from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorWithPaddingAndNesting:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # flatten batch: [ [cand1, cand2], [cand1, cand2] ] -> [cand1, cand2, cand1, cand2]
        features = [ff for f in features for ff in f]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids")

        return batch


@dataclass
class NondiffQACollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        gold = [f.get("gold", []) for f in features]
        stripped = []
        for f in features:
            row = {k: v for k, v in f.items() if k != "gold"}
            stripped.append(row)

        batch = self.tokenizer.pad(
            stripped,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids")

        batch["gold"] = gold
        return batch

