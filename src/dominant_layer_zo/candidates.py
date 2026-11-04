"""MeZO nested candidate training representation."""
from torch.utils.data import Dataset
MEZO_CANDIDATES = {
    "sst2_mezo": [" terrible", " great"], "rte_mezo": ["Yes", "No"],
    "cb_mezo": ["Yes", "No", "Maybe"], "boolq_mezo": ["Yes", "No"],
    "wsc_mezo": ["No", "Yes"], "wic_mezo": ["No", "Yes"], "multirc_mezo": ["No", "Yes"],
}

def _truncate_left_keep_bos(tokenizer, ids, max_length):
    if len(ids) <= max_length:
        return ids
    bos_id = getattr(tokenizer, "bos_token_id", None)
    add_bos = bool(getattr(tokenizer, "add_bos_token", False))
    if add_bos and bos_id is not None and len(ids) > 0 and ids[0] == bos_id and max_length >= 2:
        return ids[:1] + ids[1:][-(max_length - 1):]
    return ids[-max_length:]


class NestedCandidateDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _build_mezo_classification_dataset(ds, task_name, tokenizer, cutoff_len):
    candidates = MEZO_CANDIDATES[task_name]
    items = []

    for ex in ds:
        prompt = ex["prompt"]
        gold_text = ex["full_prompt"][len(prompt):]

        if gold_text not in candidates:
            raise ValueError(
                f"Gold suffix {gold_text!r} not found in candidates for {task_name}: {candidates}"
            )
        gold_idx = candidates.index(gold_text)

        prompt_ids = tokenizer.encode(prompt)

        nested = []
        for cand in candidates:
            full_ids = tokenizer.encode(prompt + cand)
            option_len = len(full_ids) - len(prompt_ids)

            full_ids = _truncate_left_keep_bos(tokenizer, full_ids, cutoff_len)
            option_len = max(1, min(option_len, len(full_ids) - 1))

            nested.append({
                "input_ids": full_ids,
                "labels": gold_idx,          # class id, repeated for every candidate
                "option_len": option_len,    # number of candidate tokens at the end
                "num_options": len(candidates),
            })

        items.append(nested)

    return NestedCandidateDataset(items)

