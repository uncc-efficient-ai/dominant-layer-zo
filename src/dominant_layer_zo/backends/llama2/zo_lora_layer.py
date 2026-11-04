import random
import re
import string

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer

from .lora_common import (
    assert_lora_model,
    attach_mezo_classification_forward_if_needed,
    collect_trainable_params,
    count_params,
    ensure_lora_scope,
    get_lora_layer_idx,
    get_tune_name,
    log_trainable_sample,
)


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


class MyTrainer(Trainer):
    """MeZO trainer for LoRA attached only inside one selected transformer layer."""

    def __init__(self, *args, task_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert task_config is not None, "task_config required"
        self.task_config = task_config

        zo = task_config.task.tune.zo
        self.eps = float(zo.eps)
        self.lr = float(zo.lr)
        self.directions = int(getattr(zo, "directions", 1))
        self.seed = int(getattr(zo, "seed", 0))
        self.layer_idx = get_lora_layer_idx(task_config)

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

        tune_name = get_tune_name(task_config)
        if self.non_diff and tune_name not in {"squad", "squad_mezo"}:
            raise ValueError("zo.non_diff is currently supported only for squad/squad_mezo in zo_lora_layer_trainer.")

        self.noise = str(getattr(zo, "noise", "gaussian")).lower()
        self.cast_trainable_fp32 = bool(getattr(zo, "cast_trainable_fp32", True))
        raw_max_zo_coeff = getattr(zo, "max_zo_coeff", None)
        if isinstance(raw_max_zo_coeff, dict):
            raw_max_zo_coeff = None
        self.max_zo_coeff = None if raw_max_zo_coeff is None else float(raw_max_zo_coeff)
        raw_log_zo_internals = getattr(zo, "log_zo_internals", True)
        if isinstance(raw_log_zo_internals, dict):
            raw_log_zo_internals = True
        self.log_zo_internals = bool(raw_log_zo_internals)
        self._py_rng = random.Random(self.seed)

        inferred_layer_idx = ensure_lora_scope(task_config, "single_layer", model=self.model)
        if inferred_layer_idx is not None:
            self.layer_idx = inferred_layer_idx
        assert_lora_model(self.model)
        attach_mezo_classification_forward_if_needed(self.model, task_config)

        self.trainable = collect_trainable_params(self.model, self.cast_trainable_fp32, "zo_lora_layer")
        self.trainable_dim = int(sum(p.numel() for p in self.trainable))
        self.has_fp16_trainable = any(p.data.dtype == torch.float16 for p in self.trainable)
        self._last_finite_loss = None
        self._skipped_zo_steps = 0

        trainable_count, total_count = count_params(self.model)
        print(f"[ZO-LoRA-Layer] layer_idx={self.layer_idx} noise={self.noise}")
        print(f"[ZO-LoRA-Layer] max_zo_coeff={self.max_zo_coeff} log_zo_internals={self.log_zo_internals}")
        print(f"[ZO-LoRA-Layer] Trainable params: {trainable_count} / {total_count} ({trainable_count/max(total_count,1):.6f})")
        log_trainable_sample(self.model)

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

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

    def _sample_zo_seed(self) -> int:
        return self._py_rng.randint(0, 2**31 - 1)

    def _reset_torch_rng(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _noise_like(self, p: torch.Tensor) -> torch.Tensor:
        if self.noise == "gaussian":
            return torch.randn(p.shape, device=p.device, dtype=p.dtype)
        if self.noise == "rademacher":
            return torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        raise ValueError(f"Unknown zo.noise='{self.noise}'. Use 'gaussian'|'rademacher'.")

    def _perturb(self, scale: float, seed: int):
        self._reset_torch_rng(seed)
        for p in self.trainable:
            z = self._noise_like(p)
            p.add_(z, alpha=scale)

    def _update(self, step_lr: float, coeff: torch.Tensor, seed: int):
        self._reset_torch_rng(seed)
        for p in self.trainable:
            z = self._noise_like(p)
            z.mul_(coeff)
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        model.eval()

        with torch.no_grad():
            seeds = []
            coeffs = []
            sym_loss = torch.zeros((), device=self.args.device, dtype=torch.float32)
            zo_log = {}
            invalid_step = False

            for _ in range(self.directions):
                s = self._sample_zo_seed()
                seeds.append(s)

                self._perturb(self.eps, s)
                with torch.inference_mode():
                    loss_plus = self.compute_loss_from_batch(model, inputs).detach().float()

                self._perturb(-2.0 * self.eps, s)
                with torch.inference_mode():
                    loss_minus = self.compute_loss_from_batch(model, inputs).detach().float()

                coeff = (loss_plus - loss_minus) / (2.0 * self.eps)
                raw_coeff = coeff.detach().clone()

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
                return torch.zeros((), device=self.args.device, dtype=torch.float32)

            step_lr = self.lr / float(self.directions)
            for s, coeff in zip(seeds, coeffs):
                self._update(step_lr, coeff, s)

            sym_loss = sym_loss / float(self.directions)
            if torch.isfinite(sym_loss):
                self._last_finite_loss = sym_loss.detach().float()

            if self.log_zo_internals and (self.state.global_step + 1) % logging_steps == 0:
                zo_log["zo/skipped_steps_total"] = float(self._skipped_zo_steps)
                self.log(zo_log)

            return sym_loss.detach()
