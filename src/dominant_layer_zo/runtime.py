import contextlib
import fcntl
import importlib.metadata
import json
import hashlib
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def environment():
    return {'python':platform.python_version(), 'packages':{
        x:importlib.metadata.version(x) for x in ['torch','transformers','datasets','accelerate','numpy','tokenizers','huggingface-hub','safetensors','sentencepiece']},
        'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'utc':datetime.now(timezone.utc).isoformat()}


@contextlib.contextmanager
def run_lock(output, resume=False):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {path}') from None
        existing = [x for x in path.iterdir() if x.name != '.lock']
        if existing and not resume:
            raise FileExistsError(f'{path} is nonempty; use a new output_dir or an explicit compatible checkpoint')
        yield path


def implementation_signature():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for source in sorted(root.rglob('*.py')):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()
