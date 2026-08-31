# verpex

Deep-learning prediction of anatomical points-of-interest (POIs) on vertebrae, from CT
and MRI spine segmentations.

The model works one vertebra at a time. A DenseNet backbone predicts a heatmap per
landmark on a fixed-size cutout, and a transformer then refines those coarse coordinates
using image patches taken around them, so the final prediction is sub-voxel accurate.

Built on [TPTBox](https://github.com/Hendrik-code/TPTBox) for BIDS dataset handling,
NIfTI I/O and POI containers.

## Installation

Requires Python 3.10 or newer.

```bash
conda create -n verpex python=3.10
conda activate verpex

pip install -e .
```

For the sparse-convolution backbones (`SMDenseNet`, `SMSADenseNet`) also install
`spconv` matching your CUDA version — the dense pipeline does not need it.

## Configuration

Machine-specific paths live in `config/paths.yaml`, which is git-ignored. Copy the
template and fill it in:

```bash
cp config/paths.example.yaml config/paths.yaml
```

| Key | Used for |
| --- | --- |
| `data_root` | BIDS dataset(s) to read images and annotations from |
| `cutout_root` | where `prepare-data` writes cutouts and `master_df.csv` |
| `model_root` | trained model directories and their checkpoints |
| `output_root` | evaluation and inference results |
| `tmp_root` | scratch space (defaults to `/tmp/verpex`) |

Every key can be overridden by an environment variable — `data_root` becomes
`VERPEX_DATA_ROOT`, and so on — which takes precedence over the file. A key that is
needed but unset raises a `PathConfigError` naming exactly what to set.

## Preparing data

A BIDS-like dataset is expected:

```text
dataset/
├── rawdata/…              CT or MR image
└── derivatives/…          vertebra instance mask, subregion mask, POI json
```

Whole scans do not fit in GPU memory, so each vertebra is cut out, brought to a standard
orientation and spacing, and written to disk once up front:

```bash
verpex-prepare-data --data_path $DATASET --derivatives_name derivatives --save_path $CUTOUTS
```

This writes one directory per vertebra plus a `master_df.csv` listing them. Paths in that
CSV are **relative to the cutout root**, so the file stays valid if the data moves.
It uses 8 worker processes by default (`--n_workers`), takes minutes to hours, and needs
several GB of disk.

## Training

Experiments are described by a JSON config. `configs/example_train.json` is a working
starting point; fill in `master_df` and the subject splits.

```bash
verpex-train --config configs/example_train.json
```

Components are addressed by a `"type"` string resolved through an explicit registry, so a
config names a model rather than importing one:

```json
{"type": "PatchTransformer", "params": {"n_landmarks": 35, "patch_size": 16}}
```

Registered names live in `verpex.registry` and the `*_MODULES` dicts beside each family
of components. An unknown name raises an error listing the valid ones.

Pass `--config-dir` instead to run every config in a directory in sequence, or use
`verpex-train-cv --n_folds 5` for cross-validation.

## Evaluating and predicting

```bash
verpex-eval  --checkpoint_path $CKPT --split test --project
verpex-infer --datasets $DATASET_NAME --der_msk derivatives
```

`verpex-eval` writes per-POI, per-vertebra and per-subject metric CSVs plus an outlier
list. All errors are in millimetres. `verpex-infer` runs the full pipeline from raw
masks to a BIDS POI file.

## Versioning

The version is not written in `pyproject.toml`; it is derived from the latest git tag at
build time by [poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning),
the same way TPTBox does it. `version = "0.0.0"` in `pyproject.toml` is only a placeholder.

Release by tagging:

```bash
git tag v0.1.0 && git push --tags
```

Between tags the version reads as `0.1.0.post<n>.dev0+<sha>`. **A repository with no tags
at all builds as `0.0.0.post<n>.dev0+<sha>`** — so tag once after the initial commit.

`verpex.__version__` reads the installed package metadata, so it always agrees with the
built version. It reports `0.0.0+unknown` when the source tree was never installed.

## Development

```bash
pip install -e . && pip install pytest ruff pre-commit
pre-commit install

pytest
ruff check . && ruff format --check .
python scripts/check_imports.py
```

The test suite runs on synthetic tensors and needs no dataset. `scripts/check_imports.py`
imports every module and is the quickest check that a refactor did not break the package.

A pre-commit hook rejects absolute machine paths and clinical subject identifiers; see
`scripts/check_no_private_data.py`. Please keep it passing rather than bypassing it —
this repository is derived from work on clinical data.

## Notes on this release

`CHANGES.md` records every behavioural difference from the internal research codebase this
was extracted from, including several fixes that change numerical results. Read it before
comparing against older runs.

## Exporting this branch to a new repository

Export the **working tree**, not the git history. The history of the repository this was
extracted from contains material that must not be published (see `CHANGES.md` and the
notes below), and `git archive` copies only tracked files, so nothing ignored comes along:

```bash
mkdir ../verpex && git archive clean-repo | tar -x -C ../verpex
cd ../verpex && git init && git add -A && git commit -m "Initial commit"
git tag v0.1.0    # dynamic versioning needs a tag; see Versioning above
```

Before pushing anywhere public, settle `LICENSE.TODO.md` and `AUTHORS.md`.

## Citation and provenance

This code descends from the master's thesis *Automated Point-of-Interest Prediction on CT
Scans of Human Vertebrae Using Spine Segmentations* by Daniel-Jordi Regenbrecht
([original repository](https://github.com/doppelplusungut/3dVertPois)), and a subsequent
bachelor's thesis at TUM. See `AUTHORS.md`.
