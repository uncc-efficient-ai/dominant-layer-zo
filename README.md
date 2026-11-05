# Dominant-Layer ZO

Official implementation of **Dominant-Layer ZO: A Single Layer Dominates
Zeroth-Order Fine-Tuning of LLMs**, accepted at **NeurIPS 2026**.

Activation-based layer selection and zeroth-order fine-tuning for large language
models. The workflow identifies a dominant transformer block from activation
statistics, then applies finite-difference updates to that block while freezing
the rest of the model. Full-model MeZO, SparseMeZO and AGZO provide baselines
through the same task, training and evaluation pipeline.

[Setup](#setup) · [Training](#training) · [Evaluation](#evaluation) ·
[Hyperparameters](#hyperparameters-and-reproducibility) · [Countdown ES](#countdown-es) ·
[Code structure](#code-structure)

| Model family | Hugging Face model | Supported methods |
|---|---|---|
| `llama2` | `meta-llama/Llama-2-7b-hf` | MeZO, SparseMeZO, AGZO, FO, ZO-LoRA, FO-LoRA |
| `qwen3` | `Qwen/Qwen3-8B-Base` | MeZO, SparseMeZO, AGZO, FO, ZO-LoRA, FO-LoRA |
| `mistral` | `mistralai/Mistral-7B-v0.1` | MeZO, SparseMeZO, AGZO |

Dense MeZO supports **full-model and single-layer** training on all three models.
SparseMeZO and AGZO are **full-model baselines**. FO and LoRA support both scopes
on LLaMA2 and Qwen3. The separate [Countdown ES experiment](#countdown-es) uses
`Qwen/Qwen3-8B`.

## Setup

Use Python 3.10–3.12 and a CUDA-capable GPU. Dependencies are pinned in
[pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/Brian-WY/dominant-layer-zo.git
cd dominant-layer-zo
python -m venv .venv
source .venv/bin/activate
pip install -e .
dlzo --help
```

Run the commands below from the repository root. Models and datasets download
through Hugging Face; obtain access to gated models and run `huggingface-cli login`
when needed. Use one process and one visible GPU per experiment.

ZO uses fp16 or fp32 model weights; FO uses fp32 or bf16. SparseMeZO/AGZO also
keep CPU parameter snapshots: allow about 16 GB of additional host memory for an
8B fp16 full-model run, plus memory for loading the model and datasets.

## Training

### Select and train a dominant layer

The following example measures LLaMA2 activations and trains the selected block
on COPA. Use the same model revision for measurement and training.

```bash
CUDA_VISIBLE_DEVICES=0 dlzo measure --family llama2 \
  --revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --output runs/measure-llama2

CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
  --selection runs/measure-llama2/measurement.json
```

Calibration uses 16 prompt-only training examples per COPA/WSC condition, with
seeds `0,42,2026`. Padding is excluded from activation statistics. Selection takes
the first block after block 0 whose maximum absolute activation is at least
10 times the preceding block's maximum, at least 100, and at least 1,000 times
its own mean absolute activation. `--selection` requires agreement across all
calibration conditions and a matching model/revision.

For Qwen3 or Mistral, use the corresponding `--family`, model revision from
`configs/<family>/single_layer.yaml`, and a new measurement output directory.
For a manual layer experiment, omit `--selection` and set `--set training.layer=N`.
Indices are zero-based; `training.part=layer` updates the entire selected block.

### Run full-model baselines

Full-model runs do not require activation-based layer selection:

```bash
# Dense MeZO on COPA
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/full_model.yaml

# SparseMeZO and AGZO on SST-2
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/sparse_mezo_full.yaml
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/agzo_full.yaml
```

Replace `llama2` with `qwen3` or `mistral` to use the other model presets.
The dense MeZO presets use Rademacher directions; `--set training.noise=gaussian`
selects Gaussian directions. SparseMeZO/AGZO use one Gaussian direction and
require `model.fp32_attention_scores=true`.

### Choose a task or method

Use `--set SECTION.KEY=VALUE` for overrides and a new `output_dir` for every run:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/qwen3/single_layer.yaml \
  --set data.task=sst2 --set training.layer=6 --set training.lr=1e-6 \
  --set output_dir=runs/qwen3-sst2-layer6
```

Task names: `sst2`, `rte`, `cb`, `boolq`, `wsc`, `multirc`, `copa`, `squad`,
`drop`, `wic`, `record`. Choose a configuration under `configs/<family>/`:

| Configuration | Experiment |
|---|---|
| `single_layer.yaml`, `full_model.yaml` | Dense MeZO (`training.method=zo`) |
| `sparse_mezo_full.yaml`, `agzo_full.yaml` | Full-model baselines |
| `fo_{layer,full}.yaml` | AdamW fine-tuning; LLaMA2/Qwen3 |
| `zo_lora_{layer,full}.yaml`, `fo_lora_{layer,full}.yaml` | LoRA fine-tuning; LLaMA2/Qwen3 |
| `squad_fp16_{layer,full}.yaml` | SQuAD language-model objective; LLaMA2/Mistral |
| `sst2_fp16_{layer,full}.yaml` | SST-2 dense MeZO; Mistral |

Check any training command by adding `--dry-run`; this validates settings without
loading a model, dataset or GPU.

<details>
<summary>Short execution check</summary>

Check a small training budget before a longer run:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/qwen3/agzo_full.yaml \
  --set training.steps=32 --set training.eval_steps=16 \
  --set training.logging_steps=1 --set data.loader_kwargs.dev_n=32 \
  --set evaluation.after_training=false --set checkpoint.save_final=false \
  --set output_dir=runs/qwen3-agzo-check
```

Inspect `run.json` and the loss history in `trainer_state.json`. Use the same dev
split to judge the loss trend; individual training batches can fluctuate.
SparseMeZO/AGZO report the positive-probe loss during training.

</details>

## Evaluation

Training runs downstream evaluation when `evaluation.after_training=true`.
To evaluate a base model or saved weights separately:

```bash
# Base-model evaluation
CUDA_VISIBLE_DEVICES=0 dlzo evaluate --config configs/llama2/single_layer.yaml \
  --output runs/llama2-copa-base

# Evaluate a completed training run using its resolved settings
CUDA_VISIBLE_DEVICES=0 dlzo evaluate --config runs/llama2-copa-layer/config.json \
  --checkpoint runs/llama2-copa-layer/final --output runs/llama2-copa-eval
```

Set `checkpoint.save_final=true` during training to create `final/`.
SparseMeZO, AGZO and SQuAD presets save metrics without final weights by default.
LoRA checkpoints contain adapters and load onto the matching base model.

| Run output | Contents |
|---|---|
| `config.json`, `run.json` | Resolved settings, status, model/data identity and completed steps |
| `trainer_state.json` | Training and internal-dev metric history |
| `validation.json` | Internal-dev loss for hyperparameter selection |
| `downstream_eval.json` | Final task metrics |
| `final/`, `checkpoint-*/` | Optional inference weights and restart checkpoints |

<details>
<summary>Save and resume checkpoints</summary>

Enable periodic checkpoints before starting:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
  --set checkpoint.strategy=steps --set checkpoint.save_steps=2000 \
  --set output_dir=runs/llama2-checkpoints
```

After interruption, resume from an existing checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config runs/llama2-checkpoints/config.json \
  --resume runs/llama2-checkpoints/checkpoint-2000
```

Resume uses the same output directory, settings, code, dependencies and data.
To evaluate the best dev checkpoint at the end of training, enable
`checkpoint.load_best_model_at_end=true` and make `checkpoint.save_steps` a
multiple of `training.eval_steps`.

</details>

## Hyperparameters and reproducibility

Tune on internal dev data for each model, task and method. Start with the learning
rate, then check perturbation size and training budget.

| Parameter | Role |
|---|---|
| `training.lr` | Update size; an example ZO search is `1e-7, 5e-7, 1e-6, 5e-6` |
| `training.epsilon` | ZO perturbation size; SparseMeZO/AGZO presets use `1e-3` |
| `training.steps`, `training.eval_steps` | Training budget and dev-evaluation interval |
| `training.batch_size`, `data.cutoff_len` | Batch size, sequence length and memory use |
| `training.noise`, `training.directions` | Dense ZO perturbation distribution and number of directions |
| `sparse.selected_ratio` | SparseMeZO selected fraction; default `0.25`, excluding norm parameters |
| `agzo.rank`, `agzo.power_iterations` | Activation-subspace settings; defaults `1` and `5` |
| `lora.rank`, `lora.alpha`, `lora.targets` | Adapter size, scaling and target projections |
| `fo.*` | AdamW, scheduler and gradient-accumulation settings |

For comparable runs, fix model revision, data split, task objective, forward
precision, perturbation distribution and evaluation policy. Enable
`model.fp32_attention_scores=true` on dense ZO when comparing it with
SparseMeZO/AGZO. Keep `data.seed` and `evaluation.seed` fixed; vary `training.seed`
for repeated runs.

Select an LR using otherwise identical pilot runs:

```bash
for lr in 1e-7 5e-7 1e-6 5e-6; do
  CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
    --selection runs/measure-llama2/measurement.json \
    --set training.lr="$lr" --set output_dir="runs/pilot-$lr" \
    --set evaluation.after_training=false --set checkpoint.save_final=false
done
python scripts/select_lr.py runs/pilot-1e-7 runs/pilot-5e-7 runs/pilot-1e-6 runs/pilot-5e-6 \
  --output runs/locked.yaml --run-output runs/report-seed-0
```

The selected config keeps the pilot's layer and training budget and enables
final evaluation. Reuse it across reporting seeds:

```bash
for seed in 0 42 2026; do
  CUDA_VISIBLE_DEVICES=0 dlzo train --config runs/locked.yaml \
    --set training.seed="$seed" --set output_dir="runs/report-seed-$seed"
done
python scripts/summarize.py runs/report-seed-0 runs/report-seed-42 runs/report-seed-2026
```

`select_lr.py` uses final internal-dev loss. `summarize.py` reports the mean and
sample standard deviation and requires matching settings except training seed
and output directory.

Train/dev are disjoint subsets of official training data; downstream evaluation
uses official validation. Split sizes are recorded in `run.json` and configurable
through supported `data.loader_kwargs` fields. `evaluation.max_samples` limits
final evaluation; `evaluation.dtype` controls its precision (default fp16).
Classification reports accuracy; SQuAD/DROP report token F1 and exact match.
DROP uses token-overlap scoring.

SQuAD/DROP use answer-masked training loss with `data.train_on_inputs=false`.
Increase `data.cutoff_len` if truncation leaves no supervised answer tokens.
SQuAD also supports `training.non_diff=true` with dense ZO and SparseMeZO for a
negative-F1 generation objective.

## Countdown ES

[extensions/es_countdown](extensions/es_countdown) runs the independent Countdown
experiment with `Qwen/Qwen3-8B`. Conditions are `base` (evaluation only), `full`
(all parameters), and `layer6`, `layer0`, `layer34` (one block).

```bash
python -m extensions.es_countdown.prepare_data --output data/es-countdown
python -m extensions.es_countdown.train \
  --config extensions/es_countdown/config.yaml --condition layer6 --dry-run
CUDA_VISIBLE_DEVICES=0 python -m extensions.es_countdown.train \
  --config extensions/es_countdown/config.yaml --condition layer6
```

The preset uses 200 training puzzles, 2,000 test puzzles, population 16,
16 prompts per member, 200 iterations and greedy 512-token generation.
Use `--set KEY=VALUE` to tune `es.alpha` (update size), `es.sigma` (perturbation
scale), `es.population_size` and `es.iterations`. Keep data, decoding and seeds
fixed across conditions.

`data.train` and `data.test` point to JSONL files with `prompt` (or `context`),
`numbers` and `target`. Set `data.dev` and a positive `evaluation.dev_every` for
held-out evaluation. Use a new `output_root` per study; interrupted ES runs restart
in a new output root. `save_final=true` saves inference weights. Full-model ES
needs about 16 GB of additional host memory for fp16 parameter snapshots.

## Code structure

| Path | Purpose |
|---|---|
| `src/dominant_layer_zo/` | CLI, measurement, data, training, evaluation and checkpoints |
| `src/dominant_layer_zo/backends/` | Model-specific implementations and task objectives |
| `src/dominant_layer_zo/methods/agzo.py`, `src/dominant_layer_zo/sparse.py` | AGZO and SparseMeZO implementations |
| `configs/{llama2,qwen3,mistral}/` | Model, method and task presets |
| `extensions/es_countdown/` | Countdown data preparation and ES training |
| `scripts/` | LR selection, multi-seed summaries and activation plots |
| `tests/`, `.github/workflows/` | Offline tests and CI |
| `runs/`, `data/` | Local outputs and prepared data, excluded from Git |

Plot measured activations:

```bash
pip install -e '.[plot]'
python scripts/plot_measurement.py runs/measure-llama2/measurement.json \
  --output runs/measure-llama2/activations.png
```

<details>
<summary>Run offline CPU tests</summary>

These tests use small randomly initialized models:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s extensions/es_countdown/tests -v
```

</details>

## References and acknowledgements

The baseline methods build on [MeZO](https://github.com/princeton-nlp/MeZO),
[SparseMeZO](https://github.com/NUS-HPC-AI-Lab/SparseMeZO) and
[AGZO](https://github.com/Yining-Jiang/ActivationGuidedZO-NPU). The Countdown
experiment follows [ES at Scale](https://github.com/VsonicV/es-at-scale).
Model backends use [Hugging Face Transformers](https://github.com/huggingface/transformers).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
licenses. Model weights and datasets are downloaded separately under their own terms.
