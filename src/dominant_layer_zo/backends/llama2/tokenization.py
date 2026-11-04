import logging

logger = logging.getLogger(__name__)



def tokenize(prompt, tokenizer, cutoff_len, base_model, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in base_model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in base_model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result


def generate_and_tokenize_prompt(dataset, train_on_inputs, tokenizer, cutoff_len, base_model, data_point, non_diff=False):
    # Downstream tasks: loader provides full_prompt + prompt (for masking)
    if "full_prompt" in data_point:
        full_prompt = data_point["full_prompt"]
        old_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"
        try:
            # MeZO-style non-differentiable objective path (SQuAD): train on prompt-only
            # and provide gold answers for generation-time F1 computation in the trainer.
            if non_diff and "gold_answers" in data_point:
                prompt_only = data_point.get("prompt", full_prompt)
                tokenized_prompt = tokenize(prompt_only, tokenizer, cutoff_len, base_model, add_eos_token=False)
                tokenized_prompt["gold"] = data_point["gold_answers"]
                return tokenized_prompt

            tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, base_model)
            if not train_on_inputs:
                user_prompt = data_point.get("prompt")
                if user_prompt is None:
                    # fallback: no masking if prompt not provided
                    return tokenized_full_prompt
                if dataset in {"drop", "drop_mezo"} and full_prompt.startswith(user_prompt):
                    # Both strings are left-truncated above. Tokenizing the prompt
                    # independently can therefore make user_prompt_len == cutoff_len,
                    # masking the preserved answer suffix as well. Supervise the
                    # answer suffix directly so every truncated DROP example keeps
                    # at least its answer tokens (and EOS when tokenize() appended it).
                    answer_text = full_prompt[len(user_prompt):]
                    answer_ids = tokenizer(
                        answer_text,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                    )["input_ids"]
                    supervised_len = len(answer_ids)
                    full_ids = tokenized_full_prompt["input_ids"]
                    if full_ids and full_ids[-1] == tokenizer.eos_token_id:
                        if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
                            supervised_len += 1
                    supervised_len = min(max(supervised_len, 1), len(full_ids))
                    prompt_len_after_truncation = len(full_ids) - supervised_len
                    tokenized_full_prompt["labels"] = (
                        [-100] * prompt_len_after_truncation
                        + tokenized_full_prompt["labels"][prompt_len_after_truncation:]
                    )
                else:
                    tokenized_user_prompt = tokenize(
                        user_prompt, tokenizer, cutoff_len, base_model, add_eos_token=False
                    )
                    user_prompt_len = len(tokenized_user_prompt["input_ids"])
                    tokenized_full_prompt["labels"] = (
                        [-100] * user_prompt_len
                        + tokenized_full_prompt["labels"][user_prompt_len:]
                    )
            return tokenized_full_prompt
        finally:
            tokenizer.truncation_side = old_side

    raise ValueError('Expected a MeZO prompt/full_prompt example')
