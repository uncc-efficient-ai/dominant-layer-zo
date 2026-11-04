# Dominant-Layer ZO

Measure transformer activations, select a dominant layer, and run single-layer
or full-model zeroth-order fine-tuning. Use `dlzo` for measurement, training and
downstream evaluation.

## Installation

Use Python 3.10–3.12 and a CUDA-capable GPU. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
dlzo --help
```

Dependencies are pinned in [pyproject.toml](pyproject.toml). Models and datasets
download through Hugging Face; obtain model access and run `huggingface-cli login`
when required. Each training process uses one visible GPU.

## Choose a model and training path

| Config directory | Model | Available methods |
|---|---|---|
| [configs/llama2](configs/llama2) | `meta-llama/Llama-2-7b-hf` | MeZO, FO, ZO-LoRA, FO-LoRA, SparseMeZO, AGZO |
| [configs/qwen3](configs/qwen3) | `Qwen/Qwen3-8B-Base` | MeZO, FO, ZO-LoRA, FO-LoRA, SparseMeZO, AGZO |
| [configs/mistral](configs/mistral) | `mistralai/Mistral-7B-v0.1` | MeZO, SparseMeZO, AGZO |

The following filenames are relative to the model's config directory:

| Configuration | Purpose |
|---|---|
| `full_model.yaml` | Dense MeZO over all model parameters (`method: zo`) |
| `single_layer.yaml` | Dense MeZO on one transformer block; all other parameters remain frozen |
| `sparse_mezo_full.yaml` | Full-model SparseMeZO baseline |
| `agzo_full.yaml` | Full-model AGZO baseline |
| `fo_{layer,full}.yaml` | AdamW training of one block or the full model; LLaMA2/Qwen3 |
| `zo_lora_{layer,full}.yaml` | ZO training of LoRA adapters; LLaMA2/Qwen3 |
| `fo_lora_{layer,full}.yaml` | AdamW training of LoRA adapters; LLaMA2/Qwen3 |

All three models have both dense MeZO scopes. Layer indices are zero-based;
`training.layer` selects the block and `training.part=layer` updates the whole
block. SparseMeZO and AGZO use full scope only.

The main dense MeZO examples use COPA and Rademacher directions. Set
`training.noise=gaussian` for Gaussian directions. SparseMeZO/AGZO examples use
SST-2, one Gaussian direction and fp32 attention scores. Task presets include
`squad_fp16_{layer,full}.yaml` for LLaMA2/Mistral and
`sst2_fp16_{layer,full}.yaml` for Mistral.

Supported `data.task` values: `sst2`, `rte`, `cb`, `boolq`, `wsc`, `multirc`,
`copa`, `squad`, `drop`, `wic`, `record`.

## Run an experiment

### 1. Select a layer

For a dominant-layer experiment, measure activations using the same model and
revision as the training config:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo measure --family llama2 \
  --revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --output runs/measure-llama2
```

For Qwen3/Mistral, change `--family`, `--revision` and `--output` using the
corresponding config. Default calibration uses 16 prompt-only training examples
per COPA/WSC condition, with seeds 0, 42 and 2026; padding is excluded. The selector
chooses the first block after block 0 whose maximum absolute activation reaches
10 times the preceding block's maximum, 100 in absolute value and 1,000 times its
own mean absolute activation.

`measurement.json` records each condition and its selection. Training with
`--selection` requires agreement across conditions and a matching model/revision.
For a manual layer sweep, omit `--selection` and set `training.layer` explicitly.
Full-model runs do not need layer measurement.

### 2. Validate and train

Check settings without loading a model, dataset or GPU:

```bash
dlzo train --config configs/llama2/single_layer.yaml --dry-run
```

Train the selected layer and the full-model baseline:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
  --selection runs/measure-llama2/measurement.json
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/full_model.yaml
```

Run SparseMeZO or AGZO by choosing its config:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/qwen3/sparse_mezo_full.yaml
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/qwen3/agzo_full.yaml
```

Use `--set SECTION.KEY=VALUE` to override settings. Each run needs a new
`output_dir`. For a short check with periodic internal-dev evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/qwen3/agzo_full.yaml \
  --set training.steps=32 --set training.eval_steps=16 \
  --set training.logging_steps=1 --set training.lr=1e-6 \
  --set evaluation.after_training=false --set checkpoint.save_final=false \
  --set output_dir=runs/qwen3-agzo-check
```

Inspect loss, internal-dev loss and `run.json` before extending the budget.
Judge the trend on the same dev split; individual training batches can fluctuate.
SparseMeZO/AGZO training loss is the positive-probe loss.

ZO supports fp16/fp32 parameters; FO supports fp32/bf16 parameters. SparseMeZO
and AGZO require `model.fp32_attention_scores=true`. Their CPU snapshots need
additional host memory approximately equal to the updated parameter tensors:
about 16 GB for an 8B fp16 full-model run, plus model-loading and dataset memory.

### 3. Evaluate

Training evaluates downstream tasks when `evaluation.after_training=true`.
To evaluate saved weights separately, reuse the resolved run configuration:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo evaluate \
  --config runs/llama2-copa-layer/config.json \
  --checkpoint runs/llama2-copa-layer/final --output runs/llama2-copa-eval
```

Omit `--checkpoint` to evaluate the base model. Set `checkpoint.save_final=true`
during training to create `final/`; SparseMeZO, AGZO and SQuAD presets save metrics
without final weights by default. LoRA evaluation loads adapters onto the matching
base model. `evaluation.dtype` controls final evaluation precision (default fp16).

## Hyperparameters and reproduction

Start from the selected config and tune on internal dev data for each model/task.

| Setting | What to control |
|---|---|
| `training.lr` | Update size; tune this first with a fixed dev split and training budget |
| `training.epsilon` | ZO perturbation size; check it with LR when updates are unstable or ineffective |
| `training.steps`, `training.eval_steps` | Total update budget and dev-evaluation interval |
| `training.batch_size`, `data.cutoff_len` | Examples per update, sequence length and GPU memory use |
| `training.layer`, `training.part` | Layer experiment scope; use `part=layer` for a whole-block comparison |
| `training.noise`, `training.directions` | Dense ZO perturbations; SparseMeZO/AGZO require Gaussian and one direction |
| `sparse.selected_ratio` | SparseMeZO magnitude-selection fraction; default `0.25`, with norm parameters excluded |
| `agzo.rank`, `agzo.power_iterations` | Activation-subspace size and estimation work; defaults `1` and `5` |
| `lora.rank`, `lora.alpha`, `lora.targets` | Adapter size, scaling and target projections |
| `fo.*` | AdamW settings, scheduler and gradient accumulation for FO methods |
| `evaluation.max_samples`, `evaluation.max_new_tokens` | Final evaluation size and generation budget |

Keep model revision, data split, objective, precision and evaluation settings
fixed across compared runs. Set `model.fp32_attention_scores=true` on dense ZO
when comparing it with SparseMeZO/AGZO. Keep `data.seed` and `evaluation.seed`
fixed; vary `training.seed` for repeated runs.

Select an LR from otherwise identical pilot runs, then report multiple seeds:

```bash
for lr in 1e-7 5e-7 1e-6 5e-6; do
  CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
    --selection runs/measure-llama2/measurement.json \
    --set training.lr="$lr" --set output_dir="runs/pilot-$lr" \
    --set evaluation.after_training=false --set checkpoint.save_final=false
done
python scripts/select_lr.py runs/pilot-1e-7 runs/pilot-5e-7 runs/pilot-1e-6 runs/pilot-5e-6 \
  --output runs/locked.yaml --run-output runs/report-seed-0

for seed in 0 42 2026; do
  CUDA_VISIBLE_DEVICES=0 dlzo train --config runs/locked.yaml \
    --set training.seed="$seed" --set output_dir="runs/report-seed-$seed"
done
python scripts/summarize.py runs/report-seed-0 runs/report-seed-42 runs/report-seed-2026
```

Use the same selected layer and budget in every pilot. `select_lr.py` selects
by final internal-dev loss and enables downstream evaluation in the saved config.
`summarize.py` reports the mean and sample standard deviation; reporting configs
must differ only in training seed and output directory.

Train/dev are disjoint subsets of official training data; downstream evaluation
uses official validation. Model/task-specific split sizes are recorded in `run.json`.
`data.loader_kwargs` accepts supported `train_n`, `dev_n` and `test_n` overrides;
use `evaluation.max_samples` to limit downstream evaluation.

Classification reports accuracy; SQuAD/DROP report token F1 and exact match.
DROP uses token-overlap scoring. SQuAD/DROP use answer-masked training loss with
`data.train_on_inputs=false`; increase `data.cutoff_len` if no supervised answer
tokens remain. SQuAD supports `training.non_diff=true` with dense ZO and SparseMeZO
for a generation-based negative-F1 objective.

## Outputs and checkpoint recovery

| Path inside `output_dir` | Purpose |
|---|---|
| `config.json` | Resolved settings, including overrides and selected layer |
| `run.json` | Status, model/data identity, method settings and completed steps |
| `trainer_state.json` | Training and internal-dev metric history |
| `validation.json` | Final internal-dev loss for hyperparameter selection |
| `downstream_eval.json` | Task metrics, when evaluation is enabled |
| `final/` | Inference weights or adapters, when final saving is enabled |
| `checkpoint-*/` | Model and training state for restarting an interrupted run |

Enable periodic checkpoints before starting:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config configs/llama2/single_layer.yaml \
  --set checkpoint.strategy=steps --set checkpoint.save_steps=2000 \
  --set output_dir=runs/llama2-checkpoints
```

After interruption, resume from an existing checkpoint in the same output directory:

```bash
CUDA_VISIBLE_DEVICES=0 dlzo train --config runs/llama2-checkpoints/config.json \
  --resume runs/llama2-checkpoints/checkpoint-2000
```

Resume requires matching configuration, code, dependencies and data. For best-dev
checkpoint selection, enable `checkpoint.load_best_model_at_end=true` and make
`checkpoint.save_steps` a multiple of `training.eval_steps`.

## Countdown ES

[extensions/es_countdown](extensions/es_countdown) is a separate generation-reward
experiment using `Qwen/Qwen3-8B`. Conditions are `base` (evaluation only), `full`
(all parameters), and `layer6`, `layer0`, `layer34` (one block).

```bash
python -m extensions.es_countdown.prepare_data --output data/es-countdown
python -m extensions.es_countdown.train \
  --config extensions/es_countdown/config.yaml --condition layer6 --dry-run
CUDA_VISIBLE_DEVICES=0 python -m extensions.es_countdown.train \
  --config extensions/es_countdown/config.yaml --condition layer6
```

The preset uses 200 training puzzles, 2,000 test puzzles, population 16,
16 prompts per member, 200 iterations and greedy 512-token generation. Tune
`es.alpha` (update size), `es.sigma` (perturbation scale), `es.population_size`
and `es.iterations` using `--set KEY=VALUE`; keep data and decoding settings fixed.

`data.train` and `data.test` point to JSONL files containing `prompt` (or `context`),
`numbers` and `target`. Set `data.dev` and a positive `evaluation.dev_every` for
held-out evaluation during training. Use a new `output_root` for each study;
results are grouped by condition. `save_final=true` saves inference weights.
Interrupted ES runs restart in a new output root. Full-model ES needs about 16 GB
of additional host memory for fp16 parameter snapshots.

## Repository paths and utilities

| Path | Purpose |
|---|---|
| `src/dominant_layer_zo/` | CLI, activation measurement, data, training and checkpoints |
| `src/dominant_layer_zo/backends/` | Model loading, attention, objectives and model-specific trainers |
| `src/dominant_layer_zo/methods/`, `src/dominant_layer_zo/sparse.py` | AGZO, SparseMeZO and shared ZO helpers |
| `configs/` | Model/method presets for training and evaluation |
| `extensions/es_countdown/` | Countdown data preparation and ES runner |
| `scripts/` | Background launching, LR selection, metric summaries and diagnostics |
| `tests/` | Offline checks of updates, freezing, numerics and checkpoint recovery |
| `runs/`, `data/` | Local outputs and prepared data; excluded from Git |

Launch in the background with GPU occupancy checks and saved logs:

```bash
bash scripts/launch.sh 0 llama2-copa-background train \
  --config configs/llama2/single_layer.yaml \
  --set output_dir=runs/llama2-copa-background
```

Logs, PID and exit status are under `runs/launches/<run-name>/`.
`scripts/plot_measurement.py` plots `measurement.json`; install its dependencies
with `pip install -e '.[plot]'`. `scripts/check_squad_precision.py` checks SQuAD
forward losses without training. Use `--help` for script arguments.

Run offline CPU tests with small randomly initialized models:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s extensions/es_countdown/tests -v
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and licenses.
Model weights and datasets are downloaded separately under their own terms.
