"""Task prompts and train/dev split definitions."""
from datasets import load_dataset, DatasetDict

def get_sst2_mezo(tokenizer, seed=42, train_n=1000, dev_n=500, test_n=1000):
    d = load_dataset("glue", "sst2")

    tr = d["train"].shuffle(seed=seed)
    train = tr.select(range(min(train_n, len(tr))))
    dev = tr.select(range(min(train_n, len(tr)), min(train_n + dev_n, len(tr))))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        text = ex["sentence"].strip()
        prompt = f"{text} It was"
        label = int(ex["label"])
        answer = " terrible" if label == 0 else " great"
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_rte_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "rte")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        premise = ex["premise"]
        hypo = ex["hypothesis"]
        prompt = f'{premise}\nDoes this mean that "{hypo}" is true? Yes or No?\n'
        label = int(ex["label"])
        answer = "Yes" if label == 0 else "No"
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_cb_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "cb")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def to_prompt(ex):
        premise = ex["premise"]
        hypo = ex["hypothesis"]
        prompt = f'Suppose {premise} Can we infer that "{hypo}"? Yes, No, or Maybe?\n'
        label = int(ex["label"])
        answer = verbalizer[label]
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_boolq_mezo(tokenizer, seed=42, train_n=1000, dev_n=500, test_n=1000):
    d = load_dataset("boolq")

    tr = d["train"].shuffle(seed=seed)
    train = tr.select(range(train_n))
    dev = tr.select(range(train_n, train_n + dev_n))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        passage = ex["passage"]
        question = ex["question"]
        if not question.endswith("?"):
            question += "?"
        question = question[0].upper() + question[1:]
        prompt = f"{passage} {question}\n"
        answer = "Yes" if ex["answer"] else "No"
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)

    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_wsc_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "wsc.fixed")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        text = ex["text"]
        span1 = ex["span1_text"]
        span2 = ex["span2_text"]
        prompt = (
            f'{text}\n'
            f'In the previous sentence, does the pronoun "{span2.lower()}" refer to {span1}? Yes or No?\n'
        )
        answer = "Yes" if int(ex["label"]) == 1 else "No"
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)

    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_wic_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "wic")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    verbalizer = {0: "No", 1: "Yes"}

    def to_prompt(ex):
        word = ex["word"]
        s1 = ex["sentence1"]
        s2 = ex["sentence2"]
        prompt = f'Does the word "{word}" have the same meaning in these two sentences? Yes, No?\n{s1}\n{s2}\n'
        label = int(ex["label"])
        answer = verbalizer[label]
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_multirc_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "multirc")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    verbalizer = {0: "No", 1: "Yes"}

    def to_prompt(ex):
        paragraph = ex["paragraph"]
        question = ex["question"]
        answer_text = ex["answer"]
        prompt = (
            f'{paragraph}\n'
            f'Question: {question}\n'
            f'I found this answer "{answer_text}". Is that correct? Yes or No?\n'
        )
        label = int(ex["label"])
        answer = verbalizer[label]
        return {"prompt": prompt, "full_prompt": prompt + answer}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_copa_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "copa")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def cap_first_word_like_mezo(s):
        words = s.split(" ")
        if words and words[0] != "I":
            words[0] = words[0].lower()
        return " ".join(words)

    def to_prompt(ex):
        premise = ex["premise"].rstrip()
        if premise.endswith("."):
            premise = premise[:-1]
        conj = " because " if ex["question"] == "cause" else " so "
        prompt = premise + conj
        choice1 = cap_first_word_like_mezo(ex["choice1"])
        choice2 = cap_first_word_like_mezo(ex["choice2"])
        gold = choice1 if ex["label"] == 0 else choice2
        return {"prompt": prompt, "full_prompt": prompt + gold}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)

    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_record_mezo(tokenizer, seed=42, dev_n=100, test_n=1000):
    d = load_dataset("super_glue", "record")

    tr = d["train"].shuffle(seed=seed)
    dev = tr.select(range(min(dev_n, len(tr))))
    train = tr.select(range(min(dev_n, len(tr)), len(tr)))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        passage = ex["passage"].replace("@highlight\n", "- ")
        query = ex["query"]
        answers = ex.get("answers", [])
        gold = answers[0] if isinstance(answers, list) and len(answers) > 0 else (answers or "")
        prompt = f"{passage}\n-"
        full = prompt + " " + query.replace("@placeholder", gold)
        return {"prompt": prompt, "full_prompt": full}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)
    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_squad_mezo(tokenizer, seed=42, train_n=1000, dev_n=500, test_n=1000):
    d = load_dataset("squad")

    tr = d["train"].shuffle(seed=seed)
    train = tr.select(range(train_n))
    dev   = tr.select(range(train_n, train_n + dev_n))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        title = ex["title"]
        context = ex["context"]
        question = ex["question"].strip()
        answers = ex["answers"]["text"]
        answer = answers[0] if len(answers) > 0 else ""
        prompt = f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"
        return {"prompt": prompt, "full_prompt": f"{prompt} {answer}\n", "gold_answers": answers}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)

    return DatasetDict({"train": train, "validation": dev, "test": te})


def get_drop_mezo(tokenizer, seed=42, train_n=1000, dev_n=500, test_n=1000):
    d = load_dataset("drop")

    tr = d["train"].shuffle(seed=seed)
    train = tr.select(range(min(train_n, len(tr))))
    dev = tr.select(range(min(train_n, len(tr)), min(train_n + dev_n, len(tr))))

    te = d["validation"].shuffle(seed=seed).select(range(min(test_n, len(d["validation"]))))

    def to_prompt(ex):
        passage = ex["passage"]
        question = ex["question"].strip()
        spans = ex.get("answers_spans", {}).get("spans", [])
        answer = spans[0] if spans else ""
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        return {"prompt": prompt, "full_prompt": f"{prompt} {answer}\n"}

    train = train.map(to_prompt, remove_columns=train.column_names)
    dev   = dev.map(to_prompt, remove_columns=dev.column_names)
    te    = te.map(to_prompt, remove_columns=te.column_names)

    return DatasetDict({"train": train, "validation": dev, "test": te})

