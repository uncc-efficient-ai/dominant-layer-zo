import os
import random
import math
import re
import string
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


_ZO_RNG_STATE_NAME = "zo_rng_state.pth"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _validate_zo_trainable_dtype(dtype: torch.dtype, cast_trainable_fp32: bool, trainer_name: str):
    if cast_trainable_fp32:
        return
    if dtype == torch.bfloat16:
        raise ValueError(
            f"[{trainer_name}] Raw bfloat16 trainable parameters are not supported for ZO updates. "
            "Use model.torch_dtype=float16 or float32, or cast all trainables to fp32 first."
        )
    if dtype not in {torch.float16, torch.float32}:
        raise ValueError(
            f"[{trainer_name}] Unsupported trainable dtype '{_dtype_name(dtype)}' for ZO updates. "
            "Supported low-precision mode is float16; float32 is also supported."
        )

_NO_UPCAST_CAUSAL_LOSS_DTYPE_LOGGED = False

def _causal_lm_loss_no_upcast(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, shift_labels=None, **kwargs):
    global _NO_UPCAST_CAUSAL_LOSS_DTYPE_LOGGED
    if not _NO_UPCAST_CAUSAL_LOSS_DTYPE_LOGGED:
        print(f"[ZO-Full] causal loss logits: objective={logits.dtype} upcast_fp32=False")
        _NO_UPCAST_CAUSAL_LOSS_DTYPE_LOGGED = True
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(
        logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def _validate_zo_gradient_accumulation(args, trainer_name: str):
    steps = int(getattr(args, "gradient_accumulation_steps", 1))
    if steps != 1:
        raise ValueError(
            f"[{trainer_name}] MeZO requires gradient_accumulation_steps == 1; got {steps}."
        )


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _dist_mean(x: torch.Tensor) -> torch.Tensor:
    """
    All-reduce mean for scalar tensors (loss).
    This assumes each rank computed the loss on its own mini-batch and we want the global mean.
    No effect on one GPU training. Effects for DDP only.
    """
    if not _is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= _world_size()
    return y


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_toks = _normalize_answer(prediction).split()
    gold_toks = _normalize_answer(ground_truth).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = {}
    for tok in pred_toks:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in gold_toks:
        c = common.get(tok, 0)
        if c > 0:
            num_same += 1
            common[tok] = c - 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _max_over_ground_truths(pred: str, golds) -> float:
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    return max(_f1_score(pred, g) for g in golds)


def _mezo_eos_ids(tokenizer):
    eos_ids = []
    try:
        nl = tokenizer.encode("\n", add_special_tokens=False)
        if len(nl) > 0:
            eos_ids.append(nl[-1])
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids

def forward_wrap_with_option_len(
    self,
    input_ids=None,
    labels=None,
    option_len=None,
    num_options=None,
    return_dict=None,
    **kwargs,
):
    outputs = self.original_forward(
        input_ids=input_ids,
        return_dict=return_dict,
        **kwargs,
    )

    if labels is None:
        return outputs

    logits = outputs.logits if return_dict is not False else outputs[0]

    # Shift so that tokens < n predict n
    # Models may emit fp16 logits even when Trainer AMP is disabled. Compute
    # likelihoods and CE in fp32 to avoid fp16 log-softmax/accumulation error.
    shift_logits = logits[..., :-1, :].contiguous()
    upcast_loss_logits_fp32 = bool(getattr(self, "_mezo_upcast_loss_logits_fp32", True))
    if upcast_loss_logits_fp32:
        shift_logits = shift_logits.float()
    if not bool(getattr(self, "_mezo_logged_loss_logits_dtype", False)):
        print(
            "[ZO-Full] loss logits: "
            f"model={logits.dtype} objective={shift_logits.dtype} "
            f"upcast_fp32={upcast_loss_logits_fp32}"
        )
        self._mezo_logged_loss_logits_dtype = True

    # Use input_ids, not labels, because labels may be class ids in classification mode
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()

    pad_id = getattr(self.config, "pad_token_id", None)
    if pad_id is not None:
        shift_labels[shift_labels == pad_id] = -100

    if isinstance(option_len, torch.Tensor):
        option_len = option_len.tolist()
    if isinstance(num_options, torch.Tensor):
        num_options = num_options.tolist()

    # Only compute loss on the option part
    for i, L in enumerate(option_len):
        shift_labels[i, :-int(L)] = -100

    loss_fct = CrossEntropyLoss(ignore_index=-100)

    if num_options is not None:
        # Classification over option likelihoods
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        shift_labels[~mask] = 0  # avoid bad gather indexing
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)

        if any(x != num_options[0] for x in num_options):
            # Variable-number-of-options fallback
            loss = 0.0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + int(num_options[start_id])
                _logits = selected_log_probs[start_id:end_id].unsqueeze(0)
                _labels = labels[start_id:end_id][0].unsqueeze(0)
                loss = loss + loss_fct(_logits, _labels)
                count += 1
                start_id = end_id
            loss = loss / count
        else:
            nopt = int(num_options[0])
            selected_log_probs = selected_log_probs.view(-1, nopt)
            labels = labels.view(-1, nopt)[:, 0]
            loss = loss_fct(selected_log_probs, labels)
    else:
        # LM-style option-only loss
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )

    if return_dict is False:
        output = (logits,) + outputs[1:]
        return (loss,) + output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class MyTrainer(Trainer):
    """
    MeZO-style zeroth-order trainer for FULL finetuning (updates all floating-point params).

    Per step:
      - sample integer seed s
      - reset RNG with s, perturb θ <- θ + eps*z
      - compute loss_plus
      - reset RNG with s, perturb θ <- θ - 2eps*z (now at θ - eps*z)
      - compute loss_minus
      - reset RNG with s, perturb θ <- θ + eps*z (back to θ)
      - coeff = (loss_plus - loss_minus)/(2eps)
      - reset RNG with s, update θ <- θ - lr*coeff*z

    Notes:
      - Default noise is Gaussian z~N(0,1) to match MeZO/LeZO.
      - The project entry point supports one process and one visible GPU per experiment.
    """

    def __init__(self, *args, task_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_zo_gradient_accumulation(self.args, "ZO-Full")
        assert task_config is not None, "task_config required"
        self.task_config = task_config

        zo = task_config.task.tune.zo
        self.eps = float(getattr(zo, "eps", 1e-3))
        self.lr = float(getattr(zo, "lr", 1e-6))
        self.directions = int(getattr(zo, "directions", 1))
        self.seed = int(getattr(zo, "seed", 0))

        self.mezo_classification_tasks = {
            "sst2_mezo", "rte_mezo", "cb_mezo", "boolq_mezo",
            "wsc_mezo", "wic_mezo", "multirc_mezo",
        }
        raw_non_diff = getattr(zo, "non_diff", False)
        raw_non_diff_max_new_tokens = getattr(zo, "non_diff_max_new_tokens", 32)
        raw_non_diff_sampling = getattr(zo, "non_diff_sampling", False)
        raw_non_diff_temperature = getattr(zo, "non_diff_temperature", 1.0)
        raw_non_diff_num_beams = getattr(zo, "non_diff_num_beams", 1)
        raw_non_diff_top_p = getattr(zo, "non_diff_top_p", 0.95)

        if isinstance(raw_non_diff, dict):
            raw_non_diff = False
        if isinstance(raw_non_diff_max_new_tokens, dict):
            raw_non_diff_max_new_tokens = 32
        if isinstance(raw_non_diff_sampling, dict):
            raw_non_diff_sampling = False
        if isinstance(raw_non_diff_temperature, dict):
            raw_non_diff_temperature = 1.0
        if isinstance(raw_non_diff_num_beams, dict):
            raw_non_diff_num_beams = 1
        if isinstance(raw_non_diff_top_p, dict):
            raw_non_diff_top_p = 0.95

        self.non_diff = bool(raw_non_diff)
        self.non_diff_max_new_tokens = int(raw_non_diff_max_new_tokens)
        self.non_diff_sampling = bool(raw_non_diff_sampling)
        self.non_diff_temperature = float(raw_non_diff_temperature)
        self.non_diff_num_beams = int(raw_non_diff_num_beams)
        self.non_diff_top_p = float(raw_non_diff_top_p)
        self.non_diff_top_k = getattr(zo, "non_diff_top_k", None)
        if isinstance(self.non_diff_top_k, dict):
            self.non_diff_top_k = None
        elif self.non_diff_top_k is not None:
            self.non_diff_top_k = int(self.non_diff_top_k)

        tune_name = str(task_config.task.tune.tune_dataset.name).lower()
        if self.non_diff and tune_name not in {"squad", "squad_mezo"}:
            raise ValueError("zo.non_diff is currently supported only for squad/squad_mezo in zo_full_trainer.")
        self.use_mezo_classification = (
                task_config.task.tune.tune_dataset.name in self.mezo_classification_tasks
        )
        raw_upcast_loss_logits_fp32 = getattr(zo, "upcast_loss_logits_fp32", True)
        if isinstance(raw_upcast_loss_logits_fp32, dict):
            raw_upcast_loss_logits_fp32 = True
        self.upcast_loss_logits_fp32 = bool(raw_upcast_loss_logits_fp32)
        self.model._mezo_upcast_loss_logits_fp32 = self.upcast_loss_logits_fp32
        if not self.upcast_loss_logits_fp32:
            self.model.loss_function = _causal_lm_loss_no_upcast
        raw_upcast_zo_scalar_fp32 = getattr(zo, "upcast_zo_scalar_fp32", True)
        if isinstance(raw_upcast_zo_scalar_fp32, dict):
            raw_upcast_zo_scalar_fp32 = True
        self.upcast_zo_scalar_fp32 = bool(raw_upcast_zo_scalar_fp32)
        self._logged_zo_scalar_dtype = False

        if self.use_mezo_classification and not self.non_diff:
            if not hasattr(self.model, "original_forward"):
                self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        # Noise distribution: "gaussian" (MeZO/LeZO) or "rademacher"
        self.noise = str(getattr(zo, "noise", "gaussian")).lower()

        # Optional: cast all trainable weights to fp32 (can be very expensive for 7B!)
        self.cast_trainable_fp32 = bool(getattr(zo, "cast_trainable_fp32", False))
        raw_max_zo_coeff = getattr(zo, "max_zo_coeff", None)
        if isinstance(raw_max_zo_coeff, dict):
            raw_max_zo_coeff = None
        explicit_no_coeff_clip = (
            isinstance(raw_max_zo_coeff, str)
            and raw_max_zo_coeff.strip().lower() in {"none", "null", "off", "disabled"}
        )
        self.max_zo_coeff = (
            None if raw_max_zo_coeff is None or explicit_no_coeff_clip else float(raw_max_zo_coeff)
        )
        raw_log_zo_internals = getattr(zo, "log_zo_internals", True)
        if isinstance(raw_log_zo_internals, dict):
            raw_log_zo_internals = True
        self.log_zo_internals = bool(raw_log_zo_internals)

        # Python RNG for seed sampling (we will sample on rank0 and broadcast)
        self._py_rng = random.Random(self.seed)

        # Select trainable parameters = ALL floating-point parameters
        self.trainable = []
        for p in self.model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)
                if self.cast_trainable_fp32 and p.data.dtype != torch.float32:
                    p.data = p.data.float()
                _validate_zo_trainable_dtype(p.data.dtype, self.cast_trainable_fp32, "ZO-Full")
                self.trainable.append(p)
            else:
                p.requires_grad_(False)

        self.trainable_dim = int(sum(p.numel() for p in self.trainable))
        self.has_fp16_trainable = any(p.data.dtype == torch.float16 for p in self.trainable)
        if self.has_fp16_trainable and self.max_zo_coeff is None and not explicit_no_coeff_clip:
            self.max_zo_coeff = 1000.0
        self._last_finite_loss = None
        self._skipped_zo_steps = 0

        trainable_count = sum(p.numel() for p in self.trainable)
        total_count = sum(p.numel() for p in self.model.parameters())
        print("[ZO-Full] Full finetune (all float params)")
        print(f"[ZO-Full] eps={self.eps} lr={self.lr} directions={self.directions} seed={self.seed} noise={self.noise}")
        print(
            f"[ZO-Full] upcast_loss_logits_fp32={self.upcast_loss_logits_fp32} "
            f"upcast_zo_scalar_fp32={self.upcast_zo_scalar_fp32}"
        )
        print(f"[ZO-Full] max_zo_coeff={self.max_zo_coeff} log_zo_internals={self.log_zo_internals}")
        print(f"[ZO-Full] Trainable params: {trainable_count} / {total_count} ({trainable_count/total_count:.6f})")
        print(f"[ZO-Full] Trainable dtypes: {', '.join(sorted({_dtype_name(p.dtype) for p in self.trainable}))}")

        # Ensure cache off
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        # Device for broadcasting seeds (NCCL requires CUDA tensors)
        if _is_dist() and dist.get_backend().lower() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("DDP backend is NCCL but CUDA is not available.")
            self._seed_device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._seed_device = torch.device("cpu")

    # Optimizer and scheduler interfaces required by HF Trainer.
    def create_optimizer(self):
        self.optimizer = torch.optim.SGD(self.trainable, lr=0.0)
        if not hasattr(self.optimizer, "train"):
            self.optimizer.train = lambda *args, **kwargs: None
        if not hasattr(self.optimizer, "eval"):
            self.optimizer.eval = lambda *args, **kwargs: None
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is None:
            opt = self.create_optimizer()
        self.lr_scheduler = LambdaLR(opt, lr_lambda=lambda step: 1.0)
        return self.lr_scheduler

    def optimizer_step(self, *args, **kwargs):
        return

    def lr_scheduler_step(self, *args, **kwargs):
        return

    def _save_rng_state(self, output_dir):
        super()._save_rng_state(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {"python_random_state": self._py_rng.getstate()},
            os.path.join(output_dir, _ZO_RNG_STATE_NAME),
        )

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        if checkpoint is None:
            return
        path = os.path.join(checkpoint, _ZO_RNG_STATE_NAME)
        if not os.path.isfile(path):
            print(
                f"[ZO-Full] No {_ZO_RNG_STATE_NAME} in checkpoint; "
                "continuing from the configured ZO seed."
            )
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._py_rng.setstate(state["python_random_state"])

    # ---- MeZO-style seed + RNG reset ----
    def _sample_seed_rank0_and_broadcast(self) -> int:
        """
        Sample a 31-bit integer seed on rank 0 and broadcast to all ranks.
        """
        if not _is_dist():
            return self._py_rng.randint(0, 2**31 - 1)

        s = self._py_rng.randint(0, 2**31 - 1) if _rank() == 0 else 0
        t = torch.tensor([s], device=self._seed_device, dtype=torch.int64)
        dist.broadcast(t, src=0)
        return int(t.item())

    def _reset_torch_rng(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ---- Noise sampling ----
    def _fill_noise_(self, z: torch.Tensor):
        if self.noise == "gaussian":
            # z ~ N(0, 1)
            z.normal_(mean=0.0, std=1.0)
        elif self.noise == "rademacher":
            # z in {+1, -1}
            z.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        else:
            raise ValueError(f"Unknown zo.noise='{self.noise}'. Use 'gaussian'|'rademacher'.")

    # ---- Perturb & Update ----
    def _perturb(self, scale: float, seed: int):
        """
        θ <- θ + scale * z   where z is regenerated from seed
        """
        self._reset_torch_rng(seed)
        for p in self.trainable:
            z = torch.empty_like(p)
            self._fill_noise_(z)
            p.add_(z, alpha=scale)

    def _update(self, step_lr: float, coeff: torch.Tensor, seed: int):
        """
        θ <- θ - step_lr * coeff * z   where z is regenerated from seed
        coeff is a scalar tensor (already globally consistent in DDP due to loss all-reduce).
        """
        self._reset_torch_rng(seed)

        # If you ever use model-parallel (params across multiple devices),
        # you may need per-device coeff copies. For standard single-device-per-rank, this is fine.
        for p in self.trainable:
            z = torch.empty_like(p)
            self._fill_noise_(z)
            z.mul_(coeff)            # z := coeff * z
            p.add_(z, alpha=-step_lr)

    def compute_loss_from_batch(self, model, inputs):
        if self.non_diff:
            return self.compute_nondiff_loss(model, inputs)
        outputs = model(**inputs)
        return outputs.loss

    def compute_nondiff_loss(self, model, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        gold = inputs.get("gold", None)
        if gold is None:
            raise ValueError("non_diff mode expects batch field 'gold' from collator.")

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        eos_token_id = _mezo_eos_ids(self.tokenizer)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=self.non_diff_sampling,
            temperature=self.non_diff_temperature,
            num_beams=self.non_diff_num_beams,
            top_p=self.non_diff_top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )
        if self.non_diff_top_k is not None:
            gen_kwargs["top_k"] = self.non_diff_top_k
        max_new = min(self.non_diff_max_new_tokens, max(1, int(self.model.config.max_position_embeddings) - input_ids.shape[1])) \
            if hasattr(self.model.config, "max_position_embeddings") else self.non_diff_max_new_tokens
        gen_kwargs["max_new_tokens"] = max_new

        outputs = model.generate(**gen_kwargs)
        prompt_len = input_ids.shape[1]
        preds = []
        for i in range(outputs.shape[0]):
            gen = outputs[i, prompt_len:]
            pred = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = pred.split("\n")[0].strip()
            preds.append(pred)
        f1s = [_max_over_ground_truths(preds[i], gold[i]) for i in range(len(preds))]
        return -torch.tensor(sum(f1s) / max(1, len(f1s)), dtype=torch.float32, device=input_ids.device)

    def _compute_mezo_batch_accuracy(self, model, inputs):
        if (not self.use_mezo_classification) or self.non_diff:
            return None
        if any(k not in inputs for k in ("input_ids", "labels", "option_len", "num_options")):
            return None

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        option_len = inputs["option_len"]
        num_options = inputs["num_options"]
        attention_mask = inputs.get("attention_mask", None)

        if isinstance(option_len, torch.Tensor):
            option_len = option_len.tolist()
        if isinstance(num_options, torch.Tensor):
            num_options = num_options.tolist()

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()

        pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is not None:
            shift_labels[shift_labels == pad_id] = -100

        for i, L in enumerate(option_len):
            shift_labels[i, :-int(L)] = -100

        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        shift_labels[~mask] = 0
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)

        preds, golds = [], []
        if any(x != num_options[0] for x in num_options):
            start = 0
            while start < len(num_options):
                nopt = int(num_options[start])
                end = start + nopt
                preds.append(int(torch.argmax(selected_log_probs[start:end]).item()))
                golds.append(int(labels[start].item()))
                start = end
        else:
            nopt = int(num_options[0])
            grouped_scores = selected_log_probs.view(-1, nopt)
            grouped_labels = labels.view(-1, nopt)[:, 0]
            pred_ids = torch.argmax(grouped_scores, dim=-1)
            preds = pred_ids.tolist()
            golds = grouped_labels.tolist()

        if len(golds) == 0:
            return None
        correct = sum(int(p == g) for p, g in zip(preds, golds))
        return float(correct) / float(len(golds))

    def _compute_lm_token_accuracy(self, model, inputs):
        labels = inputs.get("labels", None)
        if labels is None or (not isinstance(labels, torch.Tensor)) or labels.ndim < 2:
            return None

        forward_inputs = {k: v for k, v in inputs.items() if k != "gold"}
        with torch.inference_mode():
            logits = model(**forward_inputs).logits

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        denom = int(mask.sum().item())
        if denom <= 0:
            return None
        pred = torch.argmax(shift_logits, dim=-1)
        correct = ((pred == shift_labels) & mask).sum().item()
        return float(correct) / float(denom)

    def _compute_nondiff_batch_metrics(self, model, inputs):
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            return None
        attention_mask = inputs.get("attention_mask", None)
        gold = inputs.get("gold", None)
        if gold is None:
            return None

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        eos_token_id = _mezo_eos_ids(self.tokenizer)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            top_p=1.0,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )
        max_new = min(self.non_diff_max_new_tokens, max(1, int(self.model.config.max_position_embeddings) - input_ids.shape[1])) \
            if hasattr(self.model.config, "max_position_embeddings") else self.non_diff_max_new_tokens
        gen_kwargs["max_new_tokens"] = max_new

        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)
        prompt_len = input_ids.shape[1]

        preds = []
        for i in range(outputs.shape[0]):
            gen = outputs[i, prompt_len:]
            pred = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = pred.split("\n")[0].strip()
            preds.append(pred)

        f1s, ems = [], []
        for i in range(len(preds)):
            gold_i = gold[i] if isinstance(gold[i], list) else [gold[i]]
            f1s.append(_max_over_ground_truths(preds[i], gold_i))
            ems.append(max(1.0 if _normalize_answer(preds[i]) == _normalize_answer(g) else 0.0 for g in gold_i) if gold_i else 0.0)

        if len(f1s) == 0:
            return None
        return {
            "train_f1": float(sum(f1s) / len(f1s)),
            "train_em": float(sum(ems) / len(ems)),
        }

    def _compute_train_metrics(self, model, inputs):
        if self.non_diff:
            return self._compute_nondiff_batch_metrics(model, inputs)
        if self.use_mezo_classification:
            acc = self._compute_mezo_batch_accuracy(model, inputs)
            return {"train_acc": acc} if acc is not None else None
        acc = self._compute_lm_token_accuracy(model, inputs)
        return {"train_token_acc": acc} if acc is not None else None

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)

        # Reduce stochasticity
        model.eval()

        with torch.no_grad():
            seeds = []
            coeffs = []
            sym_loss_dtype = torch.float32 if self.upcast_zo_scalar_fp32 else self.trainable[0].dtype
            sym_loss = torch.zeros((), device=self.args.device, dtype=sym_loss_dtype)
            zo_log = {}
            invalid_step = False

            for _ in range(self.directions):
                s = self._sample_seed_rank0_and_broadcast()
                seeds.append(s)

                # θ -> θ + eps*z
                self._perturb(self.eps, s)
                with torch.inference_mode():
                    loss_plus = self.compute_loss_from_batch(model, inputs).detach()
                loss_plus = _dist_mean(loss_plus)
                if self.upcast_zo_scalar_fp32:
                    loss_plus = loss_plus.float()

                # θ+epsz -> θ-epsz via -2epsz
                self._perturb(-2.0 * self.eps, s)
                with torch.inference_mode():
                    loss_minus = self.compute_loss_from_batch(model, inputs).detach()
                loss_minus = _dist_mean(loss_minus)
                if self.upcast_zo_scalar_fp32:
                    loss_minus = loss_minus.float()

                # Match original MeZO when requested: subtract/divide in the loss dtype.
                coeff = (loss_plus - loss_minus) / (2.0 * self.eps)
                raw_coeff = coeff.detach().clone()
                if not self._logged_zo_scalar_dtype:
                    print(
                        f"[ZO-Full] ZO scalar dtypes: loss_plus={loss_plus.dtype} "
                        f"loss_minus={loss_minus.dtype} coeff={coeff.dtype}"
                    )
                    self._logged_zo_scalar_dtype = True

                # restore θ0: (θ-epsz) + epsz
                self._perturb(self.eps, s)

                finite_values = torch.isfinite(loss_plus) & torch.isfinite(loss_minus) & torch.isfinite(coeff)
                zero_fp16_loss = bool(
                    self.has_fp16_trainable and not self.non_diff and bool(((loss_plus == 0.0) | (loss_minus == 0.0)).item())
                )
                if (not bool(finite_values.item())) or zero_fp16_loss:
                    invalid_step = True
                    zo_log = {
                        "zo/loss_plus": float(loss_plus.item()) if torch.isfinite(loss_plus) else float("nan"),
                        "zo/loss_minus": float(loss_minus.item()) if torch.isfinite(loss_minus) else float("nan"),
                        "zo/coeff": float(raw_coeff.item()) if torch.isfinite(raw_coeff) else float("nan"),
                        "zo/skipped_step": 1.0,
                    }
                    break

                if self.max_zo_coeff is not None:
                    coeff = coeff.clamp(-self.max_zo_coeff, self.max_zo_coeff)
                coeffs.append(coeff)

                sym_loss = sym_loss + 0.5 * (loss_plus + loss_minus)
                zo_log = {
                    "zo/loss_plus": float(loss_plus.item()),
                    "zo/loss_minus": float(loss_minus.item()),
                    "zo/coeff": float(raw_coeff.item()),
                    "zo/coeff_clipped": float(coeff.item()),
                    "zo/skipped_step": 0.0,
                }

            logging_steps = getattr(self.args, "logging_steps", 10)
            if isinstance(logging_steps, dict):
                logging_steps = 10
            logging_steps = max(1, int(logging_steps))

            if invalid_step:
                self._skipped_zo_steps += 1
                if self.log_zo_internals and (self.state.global_step + 1) % logging_steps == 0:
                    zo_log["zo/skipped_steps_total"] = float(self._skipped_zo_steps)
                    self.log(zo_log)
                if self._last_finite_loss is not None:
                    return self._last_finite_loss.detach().clone()
                return torch.zeros((), device=self.args.device, dtype=sym_loss.dtype)

            # Apply updates at θ0
            step_lr = self.lr / float(self.directions)
            for s, coeff in zip(seeds, coeffs):
                self._update(step_lr, coeff, s)

            sym_loss = sym_loss / float(self.directions)
            if torch.isfinite(sym_loss):
                self._last_finite_loss = sym_loss.detach()

            if (self.state.global_step + 1) % logging_steps == 0:
                if self.log_zo_internals:
                    zo_log["zo/skipped_steps_total"] = float(self._skipped_zo_steps)
                    self.log(zo_log)
                metrics = self._compute_train_metrics(model, inputs)
                if metrics:
                    self.log(metrics)

        return sym_loss
