# vertpois

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
conda create -n vertpois python=3.10
conda activate vertpois

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
| `tmp_root` | scratch space (defaults to `/tmp/vertpois`) |

Every key can be overridden by an environment variable — `data_root` becomes
`VERTPOIS_DATA_ROOT`, and so on — which takes precedence over the file. A key that is
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
vertpois-prepare-data --data_path $DATASET --derivatives_name derivatives --save_path $CUTOUTS
```

This writes one directory per vertebra plus a `master_df.csv` listing them. Paths in that
CSV are **relative to the cutout root**, so the file stays valid if the data moves.
It uses 8 worker processes by default (`--n_workers`), takes minutes to hours, and needs
several GB of disk.

## Training

Experiments are described by a JSON config. `configs/example_train.json` is a working
starting point; fill in `master_df` and the subject splits.

```bash
vertpois-train --config configs/example_train.json
```

Components are addressed by a `"type"` string resolved through an explicit registry, so a
config names a model rather than importing one:

```json
{"type": "PatchTransformer", "params": {"n_landmarks": 35, "patch_size": 16}}
```

Registered names live in `vertpois.registry` and the `*_MODULES` dicts beside each family
of components. An unknown name raises an error listing the valid ones.

Pass `--config-dir` instead to run every config in a directory in sequence, or use
`vertpois-train-cv --n_folds 5` for cross-validation.

## Evaluating and predicting

```bash
vertpois-eval  --checkpoint_path $CKPT --split test --project
vertpois-infer --datasets $DATASET_NAME --der_msk derivatives
```

`vertpois-eval` writes per-POI, per-vertebra and per-subject metric CSVs plus an outlier
list. All errors are in millimetres. `vertpois-infer` runs the full pipeline from raw
masks to a BIDS POI file.

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
mkdir ../vertpois && git archive clean-repo | tar -x -C ../vertpois
cd ../vertpois && git init && git add -A && git commit -m "Initial commit"
```

Before pushing anywhere public, settle `LICENSE.TODO.md` and `AUTHORS.md`.

## Citation and provenance

This code descends from the master's thesis *Automated Point-of-Interest Prediction on CT
Scans of Human Vertebrae Using Spine Segmentations* by Daniel-Jordi Regenbrecht
([original repository](https://github.com/doppelplusungut/3dVertPois)), and a subsequent
bachelor's thesis at TUM. See `AUTHORS.md`.
