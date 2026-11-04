import json
import logging
from typing import Dict, List, Tuple, Optional
import re, string
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _get_model_input_device(model) -> torch.device:
    # Works better than model.device for device_map / wrapped models
    try:
        emb = model.get_input_embeddings()
        return emb.weight.device
    except Exception:
        for p in model.parameters():
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _truncate_left_keep_bos(tokenizer, ids: List[int], max_length: int) -> List[int]:
    if len(ids) <= max_length:
        return ids
    bos_id = getattr(tokenizer, "bos_token_id", None)
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    if add_bos and bos_id is not None and len(ids) > 0 and ids[0] == bos_id and max_length >= 2:
        # keep BOS, left-truncate remainder
        return ids[:1] + ids[1:][-(max_length - 1):]
    return ids[-max_length:]

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))

def _max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)

@torch.no_grad()
def _score_batch(
    model,
    tokenizer,
    input_ids_list: List[List[int]],
    option_lens: List[int],
    reduce: str = "mean",  # "mean" (MeZO default) or "sum" (alternative scoring)
) -> List[float]:
    device = _get_model_input_device(model)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # fallback (e.g., GPT2)
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    bsz = len(input_ids_list)
    max_len = max(len(x) for x in input_ids_list)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, ids in enumerate(input_ids_list):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :L] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (bsz, max_len, vocab)

    # Shift to get next-token logprobs
    shift_logits = logits[:, :-1, :].float()     # float() for stability
    shift_labels = input_ids[:, 1:]              # (bsz, max_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (bsz, max_len-1)

    seq_lens = attention_mask.sum(dim=1).tolist()  # unpadded lengths

    scores = []
    for i in range(bsz):
        seq_len = int(seq_lens[i])
        label_len = max(seq_len - 1, 0)
        opt_len = int(option_lens[i])

        opt_len = min(opt_len, label_len)
        if opt_len <= 0:
            scores.append(float("-inf"))
            continue

        start = label_len - opt_len
        end = label_len

        chunk = token_logp[i, start:end]
        if reduce == "sum":
            scores.append(chunk.sum().item())
        else:
            scores.append(chunk.mean().item())
    return scores

def _sst2_prompt_and_candidates(ex):
    text = ex["sentence"].strip()
    prompt = f"{text} It was"
    candidates = [" terrible", " great"]  # leading space is important
    gold_idx = int(ex["label"])
    return prompt, candidates, gold_idx


def _rte_prompt_and_candidates(ex):
    premise = ex["premise"]
    hypo = ex["hypothesis"]
    prompt = f'{premise}\nDoes this mean that "{hypo}" is true? Yes or No?\n'
    candidates = ["Yes", "No"]
    gold_idx = int(ex["label"])  # 0->Yes, 1->No
    return prompt, candidates, gold_idx


def _cb_prompt_and_candidates(ex):
    premise = ex["premise"]
    hypo = ex["hypothesis"]
    prompt = f'Suppose {premise} Can we infer that "{hypo}"? Yes, No, or Maybe?\n'
    candidates = ["Yes", "No", "Maybe"]
    gold_idx = int(ex["label"])
    return prompt, candidates, gold_idx

def _boolq_prompt_and_candidates(ex) -> Tuple[str, List[str], int]:
    passage = ex["passage"]
    question = ex["question"]
    if not question.endswith("?"):
        question = question + "?"
    question = question[0].upper() + question[1:]
    prompt = f"{passage} {question}\n"
    candidates = ["Yes", "No"]
    gold_idx = 0 if bool(ex["answer"]) else 1
    return prompt, candidates, gold_idx

def _wic_prompt_and_candidates(ex):
    word = ex["word"]
    s1 = ex["sentence1"]
    s2 = ex["sentence2"]
    prompt = f'Does the word "{word}" have the same meaning in these two sentences? Yes, No?\n{s1}\n{s2}\n'
    candidates = ["No", "Yes"]   # label 0->No, 1->Yes
    gold_idx = int(ex["label"])
    return prompt, candidates, gold_idx

def _wsc_prompt_and_candidates(ex) -> Tuple[str, List[str], int]:
    text = ex["text"]
    span1 = ex["span1_text"]
    span2 = ex["span2_text"]

    prompt = (
        f'{text}\n'
        f'In the previous sentence, does the pronoun "{span2.lower()}" refer to {span1}? Yes or No?\n'
    )
    candidates = ["No", "Yes"]
    gold_idx = int(ex["label"])  # 0->No, 1->Yes
    return prompt, candidates, gold_idx

def _multirc_prompt_and_candidates(ex):
    paragraph = ex["paragraph"]
    question = ex["question"]
    answer_text = ex["answer"]
    prompt = (
        f'{paragraph}\n'
        f'Question: {question}\n'
        f'I found this answer "{answer_text}". Is that correct? Yes or No?\n'
    )
    candidates = ["No", "Yes"]   # label 0->No, 1->Yes
    gold_idx = int(ex["label"])
    return prompt, candidates, gold_idx


def _copa_cap_first_word_like_mezo(s: str) -> str:
    words = s.split(" ")
    if words and words[0] != "I":
        words[0] = words[0].lower()
    return " ".join(words)

def _copa_prompt_and_candidates(ex) -> Tuple[str, List[str], int]:
    premise = ex["premise"].rstrip()
    if premise.endswith("."):
        premise = premise[:-1]
    conj = " because " if ex["question"] == "cause" else " so "
    prompt = premise + conj

    c1 = _copa_cap_first_word_like_mezo(ex["choice1"])
    c2 = _copa_cap_first_word_like_mezo(ex["choice2"])
    candidates = [c1, c2]
    gold_idx = int(ex["label"])  # 0 => choice1, 1 => choice2
    return prompt, candidates, gold_idx

def _squad_prompt(ex) -> str:
    title = ex["title"]
    context = ex["context"]
    question = ex["question"].strip()
    return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

def _drop_prompt(ex) -> str:
    passage = ex["passage"]
    question = ex["question"].strip()
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:"

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
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids

def _build_task(name: str):
    name = name.lower()
    if name in {"sst2", "sst2_mezo"}:
        hf_name = ("glue", "sst2")
        split = "validation"
        builder = _sst2_prompt_and_candidates
        return hf_name, split, builder
    if name in {"rte", "rte_mezo"}:
        hf_name = ("super_glue", "rte")
        split = "validation"
        builder = _rte_prompt_and_candidates
        return hf_name, split, builder
    if name in {"cb", "cb_mezo"}:
        hf_name = ("super_glue", "cb")
        split = "validation"
        builder = _cb_prompt_and_candidates
        return hf_name, split, builder
    if name in {"boolq", "boolq_mezo"}:
        hf_name = ("boolq", None)
        split = "validation"
        builder = _boolq_prompt_and_candidates
        return hf_name, split, builder
    if name in {"wsc", "wsc_fixed", "wsc.fixed", "wsc_mezo"}:
        hf_name = ("super_glue", "wsc.fixed")
        split = "validation"
        builder = _wsc_prompt_and_candidates
        return hf_name, split, builder
    if name in {"wic", "wic_mezo"}:
        hf_name = ("super_glue", "wic")
        split = "validation"
        builder = _wic_prompt_and_candidates
        return hf_name, split, builder
    if name in {"multirc", "multirc_mezo"}:
        hf_name = ("super_glue", "multirc")
        split = "validation"
        builder = _multirc_prompt_and_candidates
        return hf_name, split, builder
    if name in {"copa", "copa_mezo"}:
        hf_name = ("super_glue", "copa")
        split = "validation"
        builder = _copa_prompt_and_candidates
        return hf_name, split, builder
    raise ValueError(f"Unsupported downstream task: {name}")

def _parse_optional_int(x):
    if x is None:
        return None
    # Config systems may give dict placeholders for missing values
    if isinstance(x, dict):
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        xi = int(x)
        return xi if xi > 0 else None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        xi = int(s)
        return xi if xi > 0 else None
    return None

def evaluate_downstream_accuracy(
    model,
    tokenizer,
    task_name: str,
    max_length: int,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    (ds_name, ds_config), split, builder = _build_task(task_name)
    ds = load_dataset(ds_name, ds_config)[split]

    n_max = _parse_optional_int(max_samples)
    if n_max is not None:
        n = min(n_max, len(ds))
        if n > 0:
            ds = ds.shuffle(seed=seed).select(range(n))

    # Build all candidate sequences
    all_input_ids, all_option_lens = [], []
    meta = []  # (example_idx, candidate_idx)
    gold = []  # gold candidate index per example
    num_cands = []

    for i, ex in enumerate(ds):
        prompt, candidates, gold_idx = builder(ex)
        gold.append(gold_idx)
        num_cands.append(len(candidates))

        # Compute prompt ids once (for option_len like MeZO)
        prompt_ids = tokenizer.encode(prompt)  # add_special_tokens default (MeZO style)
        for j, cand in enumerate(candidates):
            full_text = prompt + cand
            full_ids = tokenizer.encode(full_text)
            opt_len = len(full_ids) - len(prompt_ids)

            # Left truncate to keep the option at the end
            full_ids = _truncate_left_keep_bos(tokenizer, full_ids, max_length)
            # Safety clamp
            opt_len = max(1, min(opt_len, len(full_ids) - 1))

            all_input_ids.append(full_ids)
            all_option_lens.append(opt_len)
            meta.append((i, j))

    # Score in batches
    model.eval()
    scores_flat = []
    for start in range(0, len(all_input_ids), batch_size):
        batch_ids = all_input_ids[start:start + batch_size]
        batch_lens = all_option_lens[start:start + batch_size]
        scores_flat.extend(_score_batch(model, tokenizer, batch_ids, batch_lens, reduce="mean"))

    # Aggregate per example and compute accuracy
    per_example_scores: List[List[float]] = [[] for _ in range(len(ds))]
    for (ex_i, cand_j), sc in zip(meta, scores_flat):
        per_example_scores[ex_i].append(sc)

    correct = 0
    for i in range(len(ds)):
        pred = int(np.argmax(per_example_scores[i]))
        correct += int(pred == gold[i])

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    return {"accuracy": acc, "n": float(len(ds))}

def evaluate_record_accuracy(
    model,
    tokenizer,
    max_length: int,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    ds = load_dataset("super_glue", "record")["validation"]

    n_max = _parse_optional_int(max_samples)
    if n_max is not None:
        n = min(n_max, len(ds))
        if n > 0:
            ds = ds.shuffle(seed=int(seed)).select(range(n))

    # Flatten all (example, entity) into candidate sequences
    all_input_ids = []
    all_option_lens = []
    meta = []  # (ex_i, ent_j)
    gold_idx_lists = []
    entities_per_ex = []

    for i, ex in enumerate(ds):
        passage = ex["passage"].replace("@highlight\n", "- ")
        query = ex["query"]
        entities = ex.get("entities", [])
        answers = ex.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers] if answers else []

        prompt = f"{passage}\n-"
        prompt_ids = tokenizer.encode(prompt)

        # Exact MeZO-style gold handling: answer strings -> candidate indices
        gold_indices = [entities.index(ans) for ans in answers if ans in entities]

        gold_idx_lists.append(gold_indices)
        entities_per_ex.append(entities)

        for j, ent in enumerate(entities):
            filled = query.replace("@placeholder", ent)
            full_text = prompt + " " + filled
            full_ids = tokenizer.encode(full_text)
            opt_len = len(full_ids) - len(prompt_ids)

            full_ids = _truncate_left_keep_bos(tokenizer, full_ids, max_length)
            opt_len = max(1, min(opt_len, len(full_ids) - 1))

            all_input_ids.append(full_ids)
            all_option_lens.append(opt_len)
            meta.append((i, j))

    # Batch score
    model.eval()
    scores_flat = []
    for start in range(0, len(all_input_ids), batch_size):
        batch_ids = all_input_ids[start:start + batch_size]
        batch_lens = all_option_lens[start:start + batch_size]
        scores_flat.extend(_score_batch(model, tokenizer, batch_ids, batch_lens, reduce="mean"))

    # Per-example argmax over entities
    per_example_scores = [[] for _ in range(len(ds))]
    for (ex_i, ent_j), sc in zip(meta, scores_flat):
        per_example_scores[ex_i].append((ent_j, sc))

    correct = 0
    for i in range(len(ds)):
        if not per_example_scores[i]:
            continue
        best_ent_j, _ = max(per_example_scores[i], key=lambda x: x[1])
        correct += int(best_ent_j in gold_idx_lists[i])

    acc = correct / len(ds) if len(ds) > 0 else 0.0
    return {"accuracy": acc, "n": float(len(ds))}

def _mezo_prompt_budget(max_length: int, max_new_tokens: int) -> int:
    return max(1, max_length - max_new_tokens)

@torch.no_grad()
def evaluate_squad_f1_em(
    model,
    tokenizer,
    max_length: int,
    max_new_tokens: int = 32,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    ds = load_dataset("squad")["validation"]

    n_max = _parse_optional_int(max_samples)
    if n_max is not None:
        n = min(n_max, len(ds))
        if n > 0:
            ds = ds.shuffle(seed=int(seed)).select(range(n))

    device = _get_model_input_device(model)
    model.eval()

    f1_sum = 0.0
    em_sum = 0.0
    total = 0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eos_token_id = _mezo_eos_ids(tokenizer)

    for ex in ds:
        prompt = _squad_prompt(ex)
        input_ids = tokenizer.encode(prompt)

        prompt_budget = _mezo_prompt_budget(max_length, max_new_tokens)
        input_ids = _truncate_left_keep_bos(tokenizer, input_ids, prompt_budget)

        inp = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)

        out = model.generate(
            input_ids=inp,
            attention_mask=attn,
            max_new_tokens=min(max_new_tokens, max(1, max_length - inp.shape[1])),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )

        gen = out[0, inp.shape[1]:]
        pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = pred.split("\n")[0].strip()

        golds = ex["answers"]["text"]
        f1 = _max_over_ground_truths(_f1_score, pred, golds)
        em = _max_over_ground_truths(_exact_match, pred, golds)

        f1_sum += f1
        em_sum += em
        total += 1

    return {
        "f1": f1_sum / total if total else 0.0,
        "exact_match": em_sum / total if total else 0.0,
        "n": float(total),
    }

def evaluate_drop_f1_em(
    model,
    tokenizer,
    max_length: int,
    max_new_tokens: int = 50,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    ds = load_dataset("drop")["validation"]

    n_max = _parse_optional_int(max_samples)
    if n_max is not None:
        n = min(n_max, len(ds))
        if n > 0:
            ds = ds.shuffle(seed=int(seed)).select(range(n))

    device = _get_model_input_device(model)
    model.eval()

    f1_sum = 0.0
    em_sum = 0.0
    total = 0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    eos_token_id = _mezo_eos_ids(tokenizer)

    for ex in ds:
        prompt = _drop_prompt(ex)
        input_ids = tokenizer.encode(prompt)

        prompt_budget = _mezo_prompt_budget(max_length, max_new_tokens)
        input_ids = _truncate_left_keep_bos(tokenizer, input_ids, prompt_budget)

        inp = torch.tensor([input_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)

        out = model.generate(
            input_ids=inp,
            attention_mask=attn,
            max_new_tokens=min(max_new_tokens, max(1, max_length - inp.shape[1])),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_token_id,
        )

        gen = out[0, inp.shape[1]:]
        pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = pred.split("\n")[0].strip()

        golds = ex.get("answers_spans", {}).get("spans", [])
        if not golds:
            total += 1
            continue

        f1 = _max_over_ground_truths(_f1_score, pred, golds)
        em = _max_over_ground_truths(_exact_match, pred, golds)

        f1_sum += f1
        em_sum += em
        total += 1

    return {
        "f1": f1_sum / total if total else 0.0,
        "exact_match": em_sum / total if total else 0.0,
        "n": float(total),
    }

def run_downstream_eval(
    model,
    tokenizer,
    tasks: List[str],
    max_length: int,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    max_new_tokens: int = 32,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for t in tasks:
        tl = t.lower()
        if tl in {"record", "record_mezo"}:
            metrics = evaluate_record_accuracy(
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
                seed=seed,
            )
        elif tl in {"squad", "squad_mezo"}:
            metrics = evaluate_squad_f1_em(
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                max_samples=max_samples,
                seed=seed,
            )
        elif tl in {"drop", "drop_mezo"}:
            metrics = evaluate_drop_f1_em(
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            metrics = evaluate_downstream_accuracy(
                model=model,
                tokenizer=tokenizer,
                task_name=t,
                max_length=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
                seed=seed,
            )
        results[t] = metrics
        logger.info(f"[downstream_eval] {t}: {metrics}")

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[downstream_eval] wrote {output_path}")

    return results
