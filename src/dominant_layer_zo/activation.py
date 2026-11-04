"""Forward-only activation statistics and the fixed record-jump selector."""
import hashlib
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
from datasets import load_dataset

def get_transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "layers")
    ):
        return model.base_model.model.layers
    raise AttributeError("Cannot find transformer blocks at model.model.layers.")


def build_prompt_only_inputs(
    dataset_name: str,
    n_samples: int,
    seed: int,
) -> tuple[List[str], Dict[str, Any]]:
    name = dataset_name.lower()
    if name in {"copa", "copa_mezo"}:
        config = "copa"
    elif name in {"wsc", "wsc_mezo"}:
        config = "wsc.fixed"
    else:
        raise ValueError(f"Unsupported dataset={dataset_name}; use copa_mezo or wsc_mezo.")

    raw_dataset = load_dataset("super_glue", config)["train"]
    raw_fingerprint = str(raw_dataset._fingerprint)
    selected = raw_dataset.shuffle(seed=seed)
    selected = selected.select(range(min(n_samples, len(selected))))
    texts: List[str] = []
    example_ids: List[int] = []
    for example in selected:
        example_ids.append(int(example["idx"]))
        if config == "copa":
            premise = example["premise"].rstrip()
            if premise.endswith("."):
                premise = premise[:-1]
            conjunction = " because " if example["question"] == "cause" else " so "
            # No answer choice or gold label is appended.
            texts.append(premise + conjunction)
        else:
            texts.append(
                f'{example["text"]}\n'
                f'In the previous sentence, does the pronoun '
                f'"{example["span2_text"].lower()}" refer to '
                f'{example["span1_text"]}? Yes or No?\n'
            )

    provenance = {
        "dataset_builder": "super_glue",
        "dataset_config": config,
        "raw_dataset_fingerprint": raw_fingerprint,
        "selected_subset_fingerprint": str(selected._fingerprint),
        "example_ids_in_order": example_ids,
        "subset_method": "first N rows after datasets.Dataset.shuffle(seed)",
    }
    return texts, provenance


def prompt_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


class AbsActivationStat:
    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.count = 0
        self.abs_max = 0.0

    def update(self, activation: torch.Tensor, token_mask: Optional[torch.Tensor]) -> None:
        absolute = activation.detach().float().abs()
        if token_mask is not None and absolute.ndim >= 2:
            mask = token_mask.to(absolute.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(absolute.shape[:2]):
                raise ValueError(
                    f"Mask shape {tuple(mask.shape)} does not match "
                    f"activation prefix {tuple(absolute.shape[:2])}."
                )
            valid = absolute.masked_select(mask.unsqueeze(-1).expand_as(absolute))
        else:
            valid = absolute.reshape(-1)

        if valid.numel() == 0:
            return
        if not bool(torch.isfinite(valid).all().item()):
            raise FloatingPointError("Encountered a non-finite block activation.")
        self.count += int(valid.numel())
        self.sum_abs += float(valid.sum(dtype=torch.float64).item())
        self.abs_max = max(self.abs_max, float(valid.max().item()))

    def finalize(self) -> Dict[str, float]:
        return {
            "count": int(self.count),
            "abs_mean": self.sum_abs / max(1, self.count),
            "abs_max": float(self.abs_max),
        }


class LayerCollector:
    def __init__(self, model: nn.Module) -> None:
        self.layers = get_transformer_layers(model)
        self.stats = [AbsActivationStat() for _ in self.layers]
        self.handles = []
        self.token_mask: Optional[torch.Tensor] = None

        for layer_idx, layer in enumerate(self.layers):
            def make_hook(index: int):
                def hook(_module, _inputs, output):
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    self.stats[index].update(value, self.token_mask)

                return hook

            self.handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    def set_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self.token_mask = attention_mask

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rows(self) -> List[Dict[str, Any]]:
        return [
            {"layer_idx_zero_based": layer_idx, **stat.finalize()}
            for layer_idx, stat in enumerate(self.stats)
        ]


def apply_selection_rule(
    rows: Sequence[Dict[str, Any]],
    jump_threshold: float,
    min_abs_max: float,
    min_max_to_mean: float,
) -> Dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("At least two transformer layers are required.")

    enriched = []
    first = rows[0]
    first_abs_max = float(first["abs_max"])
    first_abs_mean = float(first["abs_mean"])
    enriched.append(
        {
            **first,
            "max_to_mean_ratio": first_abs_max / max(first_abs_mean, 1e-30),
            "previous_record_abs_max": None,
            "record_jump_ratio": None,
            "is_new_record": True,
            "passes_absolute_magnitude": first_abs_max >= min_abs_max,
            "passes_typical_scale": (
                first_abs_max / max(first_abs_mean, 1e-30) >= min_max_to_mean
            ),
            "qualifies": False,
            "in_candidate_set": False,
        }
    )

    previous_record = first_abs_max
    selected = None
    for row in rows[1:]:
        current = float(row["abs_max"])
        abs_mean = float(row["abs_mean"])
        denominator = max(previous_record, 1e-30)
        jump = current / denominator
        max_to_mean = current / max(abs_mean, 1e-30)
        is_new_record = current > previous_record
        passes_absolute = current >= min_abs_max
        passes_typical = max_to_mean >= min_max_to_mean
        qualifies = bool(
            is_new_record
            and jump >= jump_threshold
            and passes_absolute
            and passes_typical
        )
        layer_idx = int(row["layer_idx_zero_based"])
        if selected is None and qualifies:
            selected = layer_idx
        enriched.append(
            {
                **row,
                "max_to_mean_ratio": max_to_mean,
                "previous_record_abs_max": previous_record,
                "record_jump_ratio": jump,
                "is_new_record": is_new_record,
                "passes_absolute_magnitude": passes_absolute,
                "passes_typical_scale": passes_typical,
                "qualifies": qualifies,
                "in_candidate_set": True,
            }
        )
        previous_record = max(previous_record, current)

    ranked_jumps = sorted(
        (
            {
                "layer_idx_zero_based": int(row["layer_idx_zero_based"]),
                "record_jump_ratio": float(row["record_jump_ratio"]),
                "max_to_mean_ratio": float(row["max_to_mean_ratio"]),
                "abs_max": float(row["abs_max"]),
                "qualifies": bool(row["qualifies"]),
            }
            for row in enriched[1:]
        ),
        key=lambda row: row["record_jump_ratio"],
        reverse=True,
    )
    qualifying_layers = [
        int(row["layer_idx_zero_based"])
        for row in enriched
        if row["qualifies"]
    ]
    return {
        "formula": (
            "A_l=max_{x in C, valid token t, coordinate d}|h_l(x)[t,d]|; "
            "M_l=mean_{x,t,d}|h_l(x)[t,d]|; "
            "J_l=A_l/max_{j<l}A_j; R_l=A_l/M_l"
        ),
        "candidate_layers": "zero-based transformer blocks 1..L-1; block 0 is the baseline",
        "jump_threshold": float(jump_threshold),
        "min_abs_max": float(min_abs_max),
        "min_max_to_mean": float(min_max_to_mean),
        "tie_break": "smallest zero-based layer index",
        "fallback": "abstain if no candidate satisfies all three thresholds",
        "selected_layer_zero_based": selected,
        "selected_layer_one_based": None if selected is None else selected + 1,
        "qualifying_layers_zero_based": qualifying_layers,
        "abstained": selected is None,
        "ranked_record_jumps": ranked_jumps,
        "layers": enriched,
    }

