# Third-party notices

## Model implementations

The LLaMA2 and Qwen3 implementations in `src/dominant_layer_zo/backends/` derive
from Hugging Face Transformers with local TellMate adaptations. Their copyright
and Apache-2.0 headers are retained. The license text is included in
[licenses/transformers-Apache-2.0.txt](licenses/transformers-Apache-2.0.txt).
Adaptations include causal-language-model task support and optional fp32
attention-score arithmetic.

The Mistral backend subclasses the Transformers Mistral implementation. Its
optional attention function follows the upstream eager interface; GQA, RoPE,
sliding-window masks and cache handling use the upstream implementation.

## Training methods

Training, task and evaluation code originates from the author's TellMate project.
The optimizer, prompting and scoring conventions follow its MeZO-style
implementation. No repository-wide license has been specified for the original
project code. Third-party licenses apply to their respective components.

Custom LoRA updates follow the project's implementation inspired by
[MeZO](https://github.com/princeton-nlp/MeZO). SparseMeZO's estimator and masking
are checked against the public
[SparseMeZO trainer](https://github.com/NUS-HPC-AI-Lab/SparseMeZO/blob/0c7a92858943752117264307ece4b03209109b37/trainer.py),
commit `0c7a92858943752117264307ece4b03209109b37` (Apache-2.0 source header).

AGZO and Countdown ES implement the algorithms in
[Activation-Guided Zeroth-Order Optimization](https://arxiv.org/abs/2601.17261v1)
and [Qiu et al.](https://arxiv.org/abs/2509.24372v1).
AGZO follows the public trainer's subspace construction and rank normalization.
Reference implementations inspected were:

- `Yining-Jiang/ActivationGuidedZO-NPU`, commit
  `623e09a5a7dfdee697a603ec1e7aa01e3f5e6290`, under GPL-3.0.
- `VsonicV/es-at-scale`, commit
  `574a9d134da1ffce2a8bb812019899e5c96b588a`, under its custom noncommercial source
  license.

Those reference source files are not vendored. Method and numerical settings
are described in [README.md](README.md). The numerical regression fixture at
`tests/fixtures/reference_steps.json` contains generated inputs and outputs from
the pinned AGZO and SparseMeZO trainers, with source URLs recorded in the fixture.

## Models and data

Model weights and datasets are obtained separately and retain their own terms.
The optional Countdown exporter downloads data from the pinned reference source
and records its provenance. Model weights and datasets are not bundled.
