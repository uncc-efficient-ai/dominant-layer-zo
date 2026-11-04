"""Export the pinned upstream 200-train/2000-test Countdown data as JSONL."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from .countdown import validate_row, check_disjoint

COMMIT = "574a9d134da1ffce2a8bb812019899e5c96b588a"
ROOT = f"https://raw.githubusercontent.com/VsonicV/es-at-scale/{COMMIT}/"
FILES = {
    "train": "datasets/train/countdown/train/data-00000-of-00001.arrow",
    "test": "datasets/evaluation_suite/countdown/countdown_eval/data-00000-of-00001.arrow",
}


def prepare(output):
    import pyarrow as pa
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new data directory")
    datasets, provenance = {}, {}
    for split, path in FILES.items():
        raw = urllib.request.urlopen(ROOT + path, timeout=60).read()
        # Read IPC directly: upstream uses newer HF feature metadata.
        table = pa.ipc.open_stream(io.BytesIO(raw)).read_all()
        datasets[split] = [validate_row(row) for row in table.to_pylist()]
        provenance[split] = {"source": ROOT + path, "sha256": hashlib.sha256(raw).hexdigest(),
                             "examples": len(datasets[split])}
    if len(datasets["train"]) != 200 or len(datasets["test"]) != 2000:
        raise ValueError("Pinned upstream split sizes changed unexpectedly")
    check_disjoint(datasets["train"], datasets["test"])
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in datasets.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "source.json").write_text(json.dumps({
        "upstream_commit": COMMIT, "splits": provenance,
        "historical_rebuttal_identity": "not established"}, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
