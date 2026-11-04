"""Task construction without shared pickle caches or implicit split changes."""
import inspect
from .models import backend
from .candidates import MEZO_CANDIDATES, _build_mezo_classification_dataset
from .collators import DataCollatorWithPaddingAndNesting, NondiffQACollator
from transformers import DataCollatorForSeq2Seq


def make_data(c, tokenizer):
    d = c['data']; task = d['task'] + '_mezo'
    loader = getattr(backend(c['model']['family'], 'tasks'), 'get_' + task)
    unsupported = set(d['loader_kwargs']) - set(inspect.signature(loader).parameters)
    if unsupported:
        raise ValueError(f'{c["model"]["family"]}/{task} does not accept {sorted(unsupported)}')
    raw = loader(tokenizer=tokenizer, seed=d['seed'], **d['loader_kwargs'])
    tokenize = backend(c['model']['family'], 'tokenization').generate_and_tokenize_prompt
    info = {'task':task, 'seed':d['seed'], 'loader_defaults':{
        k:v.default for k,v in inspect.signature(loader).parameters.items()
        if k not in {'tokenizer','seed'}}, 'loader_overrides':d['loader_kwargs'],
        'split_protocol': 'train/dev are disjoint subsets of official train; test is official validation',
        'splits':{k:{'n':len(v),'fingerprint':v._fingerprint} for k,v in raw.items()}}
    if not len(raw['train']) or not len(raw['validation']):
        raise ValueError('Training and internal dev splits must both be nonempty')
    if task in MEZO_CANDIDATES:
        train = _build_mezo_classification_dataset(raw['train'],task,tokenizer,d['cutoff_len'])
        dev = _build_mezo_classification_dataset(raw['validation'],task,tokenizer,d['cutoff_len'])
        collator = DataCollatorWithPaddingAndNesting(tokenizer,pad_to_multiple_of=8)
    else:
        def encode(split):
            # Shuffle train/dev data deterministically with the configured seed.
            split = split.shuffle(seed=d['seed'])
            rows = []
            for ex in split:
                row = tokenize(task,d['train_on_inputs'],tokenizer,d['cutoff_len'],c['model']['name'],
                               ex,non_diff=c['training']['non_diff'])
                if not c['training']['non_diff'] and all(x == -100 for x in row['labels'][1:]):
                    raise ValueError('Tokenization produced no supervised answer tokens. Increase data.cutoff_len; '
                                     'see README.md for tokenization and objective settings.')
                rows.append(row)
            return rows
        train, dev = encode(raw['train']), encode(raw['validation'])
        collator = (NondiffQACollator(tokenizer) if c['training']['non_diff'] else
                    DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8,return_tensors='pt',padding=True))
    return train, dev, collator, info
