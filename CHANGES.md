# Changes from the original research codebase

This repository is a cleaned extraction of the internal `3dVertPois` research tree.
This file records every change that can alter behaviour or results, so runs made with
the original code can be re-validated. Pure refactoring — moves, renames, docstrings,
import rewrites — is not listed here.

## Fixes that change behaviour

### `RefinementModule` logging hooks raised `TypeError`

`RefinementModules.py` called `self.log_dict(metrics, on_epch=True, ...)` in both
`training_step` and `validation_step`. `on_epch` is a typo; Lightning's `log_dict`
accepts no such parameter, so either hook would raise `TypeError` on its first call.
They never did, because refinement modules run as submodules of `PoiPredictionModule`
and their own Lightning hooks are never invoked.

**Now:** spelled `on_epoch=True`. If you ever run a refinement module standalone, it
will now log instead of crashing. No effect on any existing training run.

### The prediction module named in a config was ignored

`train.py` and `train_cv.py` hard-coded `PoiPredictionModule(**config["module_config"]["params"])`,
ignoring `module_config["type"]`. A config requesting `PoiNeighborPredictionModule`
silently trained the single-vertebra module instead.

**Now:** the type string is dispatched through `PREDICTION_MODULES`. All 98 existing
configs specify `"PoiPredictionModule"`, so none of them change behaviour — but a
config that asked for the neighbour module now gets it. **If you have results from a
config naming `PoiNeighborPredictionModule`, they were produced by the wrong class.**

### The ablation refinement variants were three code generations behind

`RefinementModules.py` held eight near-duplicate transformer classes (1,503 lines,
82-91% pairwise line overlap). They were not variations on one implementation: only
`PatchTransformer` had received the recent correctness work, and the other seven had
silently fallen behind in three ways.

| | `PatchTransformer` | the 7 ablation variants |
|---|---|---|
| coarse coordinates | float (sub-voxel) | `.long()` - **truncated to whole voxels** |
| coarse features | detached | not detached - refiner gradients reach the encoder |
| loss units | millimetres (`* zoom`) | 3 of 7 optimised in **voxels**, ignoring spacing |

Each of these makes an ablation incomparable with the model it is being compared
against. The voxel-space loss is the most serious: on anisotropic data those three
variants were optimising a different objective, and their reported errors are in
different units.

**Now:** one `PatchTransformer` with `use_poi_embedding` / `use_vert_embedding` /
`use_coarse_features` / `use_patches` / `use_coarse_pred` flags, 1,503 lines down to
399. Every old class name remains a valid config `type` string, bound to the flag
combination that reproduces its architecture.

Verified by transferring weights between the old and new implementations and
comparing forward passes on synthetic input:

- **Architecture is identical for all eight names** - matching `state_dict` keys and
  shapes, so existing checkpoints still load.
- **`PatchTransformer` and `NoCoarsePredTransformer` are bit-identical.** The main
  model is unchanged.
- The other six now differ by **less than 1.0 voxel** - exactly the truncation bound
  of the `.long()` cast they used to apply, confirming the delta is the sub-voxel fix
  and nothing else.

**If you have published ablation numbers, they were produced by the three-way-divergent
code above and need re-running.** `PatchTransformer` results are unaffected.

### Unknown callbacks were silently dropped

`create_callbacks` matched `ModelCheckpoint` and `EarlyStopping` with no `else`
branch. A typo in `callbacks_config` produced a run with no checkpointing and no
early stopping, with no warning.

**Now:** an unregistered callback type raises `UnknownTypeError`. A config with a
typo that previously ran to completion (badly) will now fail immediately.

### Errors during cutout creation were masked

In `prepare_data.process_container`, the `except` block printed `crop` and `padding`.
When the failure happened at the preceding `np_center_of_mass` call, those names were
unbound, so the handler raised `UnboundLocalError` and the `raise` below it never ran —
destroying the original exception.

**Now:** both names are bound to `None` first, so the real exception propagates.
Failures that were previously reported as `UnboundLocalError` will now show their
actual cause.

### `master_df.csv` stores relative cutout paths

`prepare_data` wrote absolute `file_dir` values, and `PoiDataset` patched them at read
time by substituting a hard-coded absolute prefix — which only worked on one machine.

**Now:** `prepare_data` writes `file_dir` relative to the cutout root, and
`resolve_cutout_dir` resolves it against the configured `cutout_root`. **Absolute
paths in existing `master_df.csv` files are still honoured**, so old CSVs keep
working; regenerate them (or leave them) as you prefer.

### `SurfaceDistanceLoss` returned a norm, not a mean

It computed `torch.norm(dist.mean(dim=0), dim=0)`, which is the Euclidean norm of the
per-landmark mean vector, not the mean distance its docstring promises. The two agree
only for a single landmark; for 35 landmarks the term was inflated by roughly
`sqrt(35)` ≈ 5.9x.

**Now:** a true masked mean, and it honours `mask` like every other loss.
**If you trained with `"SD"` in a `CompoundLoss`, its weight relative to the other
terms was effectively ~6x what the config said.**

### `CompoundLoss` rejected ordinary weights, and its default contradicted its own check

`assert sum(weights) == 1.0` used exact float equality, so the perfectly reasonable
`[0.7, 0.2, 0.1]` raised (`sum` is `0.9999999999999999` in binary floating point).
Meanwhile the *default* weights were `[1.0] * n`, which sum to `n`, not 1 — the check
was enforced only on explicit weights.

**Now:** the sum is checked to a 1e-6 tolerance and raises `ValueError` (not
`AssertionError`, which `python -O` strips), and the default is `1/n` each.

### Neighbour flip augmentation mixed landmarks between vertebrae

Two separate defects, both now fixed.

`LandMarksRandHorizontalFlipNeighbor` remaps landmarks within each vertebra's block
after mirroring the volume, and the block boundaries were hardcoded as
`slice(0, 35)`, `slice(35, 70)`, `slice(70, None)` — i.e. exactly 35 landmarks per
vertebra. At any other count the blocks misaligned, and it did so **silently**: the
remapping still produced a list of the right length, so nothing raised. With
`include_com=True` (45 landmarks per vertebra, 135 total), 20 landmarks were
duplicated, 20 were dropped, and 20 were attributed to the wrong vertebra.

**Now:** the block size is derived as `len(target_indices) // n_vertebrae`, and a
landmark count that does not divide evenly raises rather than misaligning.

Separately, `GruberNeighborDataModule.__init__` printed
`"WARNING: flip_prob set to X for neighbor dataset"` and then did nothing — the line
that would have disabled flipping was commented out, and the call site carried a
`# Explizit deaktiviert` comment that was simply untrue. The base class defaults
`flip_prob=0.5`, so **any neighbour config that did not explicitly set `flip_prob: 0.0`
was training with the broken flip active.**

**Now:** the neighbour module defaults to `flip_prob = 0.0`, matching what
`PoiNeighborDataset` itself defaults to and what the reference neighbour configs set
explicitly; an explicit setting in a config is honoured rather than warned about.
**A neighbour config that omitted `flip_prob` previously flipped and now does not** —
if you want flipping, set it explicitly, and it will now be correct.

### The sparse coarse backbones computed loss and metrics in voxels

`SMDenseNet` and `SMSADenseNet` compared coordinates directly, while `SADenseNet` and
`HeatmapFeatureDenseNet` scale to millimetres first. Their loss therefore optimised a
different objective on anisotropic data, and their `coarse_mean_distance_*` metrics
were in voxels while every number logged beside them was in millimetres.

Separately, `HeatmapFeatureDenseNet`, `SMDenseNet` and `SMSADenseNet` passed `None`
where the loss expects the surface, so a `SurfaceDistanceLoss` term compounded into
their loss silently contributed zero. In `HeatmapFeatureDenseNet` the surface was also
bound only inside the `if self.project_gt:` branch, so reading it unconditionally
would have raised `UnboundLocalError` — it is now hoisted, as in `SADenseNet`.

**Now:** all four coarse modules scale to millimetres and pass the surface through.
`SADenseNet` — the only one any historical config uses — is untouched.

### Patch extraction truncated instead of rounding

`PatchExtractor.extract_patches` cast centres with `.long()`, which truncates toward
zero, biasing every patch up to a voxel toward the origin relative to the sub-voxel
coarse prediction it is meant to be centred on — while its docstring said "rounded".

**Now:** `.round().long()`. Patch contents shift by up to one voxel, so this changes
refinement results slightly.

### `--set_zoom` could not express non-integer voxel spacing

The argument parsed with `int`, so `--set_zoom 0.8,0.8,0.8` failed before preprocessing
started — on exactly the anisotropic data the coordinate-space handling is written to
get right.

**Now:** parsed as `float`.

## Fixed (previously broken, no result changes)

### Per-vertebra loss weighting now works

`PoiNeighborPredictionModule`'s `current_weight` / `neighbor_weight` had never had any
effect. Every weighted path was gated on a `"n_vertebrae"` batch key that no dataset
produced, so all of them fell through to a flat loss over the concatenated landmarks,
and the per-vertebra metrics always returned `{}`.

`PoiNeighborDataset` now emits `n_vertebrae` alongside the `n_pois_per_vertebra` it
already provided.

The implementation changed shape rather than just being switched on. The original
sliced the batch into per-vertebra blocks and called each loss once per block per
sample, which had two problems:

- **NaN.** A missing neighbour — a vertebra at either end of the spine, or one dropped
  by `neighbor_drop_prob` — has its whole block masked out, and a mean over an empty
  selection is NaN. That is routine, not an edge case, so the first such sample would
  have poisoned the batch loss.
- **Cost.** With `project_gt` enabled (78 of the historical configs) it would have run
  one surface projection per block per sample instead of one per batch.

The weights are now applied as per-landmark weights in a single loss call. Every loss
reduces through one shared `masked_weighted_mean`, which returns 0 rather than NaN for
an empty selection and keeps the autograd graph connected.

Verified: equal weights reproduce the flat loss **exactly**; `neighbor_weight=0`
reproduces the loss over the current block alone; unweighted calls are **bit-identical**
to the previous implementation, so no existing run shifts. No historical config uses
this module, so nothing you have run is affected.

`_extract_vertebra_batch` is gone with the block loops — it also never copied `"zoom"`
into its sub-batch, so it would have raised `KeyError: 'zoom'` the moment it became
reachable.

The per-vertebra metrics it enables (`current_vertebra_mean_distance_*`,
`neighbor_vertebrae_mean_distance_*`, `neighbor_to_current_distance_ratio_*`) now
report **millimetres**; they computed raw voxel distances, which would not have been
comparable with the `fine_mean_distance_*` metrics logged beside them.

### `warmup_epochs` crashed on the first training step

`PatchTransformer` is a plain `nn.Module`, so Lightning never gave it a
`current_epoch`, but the warmup branch read `self.current_epoch`. Any config with
`warmup_epochs > 0` raised `AttributeError` immediately. All 98 historical configs set
`-1`, so no completed run is affected — the feature simply never worked.

**Now:** the parent module propagates its epoch each forward pass. `current_epoch` is a
plain attribute, so it does not appear in `state_dict` and checkpoints are unchanged.

### The U-Net backbone could not be constructed

`feature_extraction.py` lazily imported `from models.DenseNet import UNetHeatmapDenseNet`
— a path from before the package move; there is no top-level `models` package.
`{"backbone": "unet"}` raised `ModuleNotFoundError`. Fixed to `verpex.models.densenet`.

### `spconv` was documented as optional but was mandatory

`modules/feature_extraction.py` imported `models/subm_densenet.py` at module scope, and
that module subclasses `spconv` types at class-definition time. Since
`feature_extraction` is reachable from every entry point, a default install (which does
not include `spconv`) could not import the package at all.

**Now:** the import is deferred into the two sparse backbones that need it, so
`spconv` is genuinely optional and CI can install without CUDA-specific wheels.

## Known-broken, left as-is (flagged, not fixed)

### Four sweep configs reference features that were never implemented

`sweep2/A1CF.json` and `no-gt-project-reproduce/surface-no_freeze-excel-A1F.json` pass
`use_ema`, and `sweep2/A1CH.json` and `...-A1H.json` ask for an `AdaptiveWingLoss`.
Neither exists anywhere in the codebase, in any commit — these configs have never been
runnable. They are not shipped with this repository; noted in case you expected results
from them. The other 94 build cleanly.

### The data modules' cutout builder uses a different cropping strategy

`POIDataModule.build_cutouts` (formerly `prepare_data`) crops each vertebra to its
bounding box plus a 5-voxel margin, while `verpex-prepare-data` writes fixed
128x128x144 cutouts. The two are not interchangeable, and only the CLI is reachable
from any entry point. It was renamed because as `prepare_data` it overrode PyTorch
Lightning's own no-argument `prepare_data()` hook: passing a data module to
`Trainer.fit(datamodule=...)` would have made Lightning call it with no arguments.

## Removed

### `--save-predictions` in `train_cv.py` (self-training pseudo-labels)

`train_cv.py` imported `create_self_training_pois` from `eval.py`, but that function
was deleted in commit `4632148` ("added zoom consideration to model and eval.py")
without updating the call site. **`train_cv.py` has raised `ImportError` on every
invocation since that commit** — the entry point was entirely broken.

The function predates that zoom-correctness pass, so it is not restored here rather
than reintroduced unverified. `train_cv.py` now works; passing `--save-predictions`
raises `NotImplementedError` with an explanation. Restoring the feature means
re-deriving it against the current zoom/offset conventions in
`verpex.geometry.spaces.revert_poi_to_original_space`.

### `L1LossMasked` and `L2LossMasked` crashed without a mask

Both assigned the reduced loss only inside `if mask is not None:` and then returned
it unconditionally, so any call that omitted the optional `mask` raised
`UnboundLocalError`.

**Now:** an unmasked call returns the mean over all landmarks. No effect on the
training path, which always passes a mask.

### A stale `.gitignore` pattern hid the `data` package

The initial `.gitignore` had a bare `data/` entry meant for datasets. Git patterns
without a leading slash match at any depth, so it also matched
`src/verpex/data/` — the package holding `dataset.py`, `transforms.py` and
`dataloading.py`. Those files stayed tracked only because they predated the rule, but
ruff (which honours `.gitignore`) silently skipped 2,600 lines of source, and any new
file added there would not have been committed.

**Now:** every dataset-shaped pattern is anchored to the repository root (`/data/`,
`/results/`, `/logs/`, …). Worth knowing about if you copy the `.gitignore` elsewhere.

### Unused surface-projection variants

`surface_project_coords_sdf_continuous`, `surface_project_coords_center_raycast`,
`surface_project_coords_voxel_boundary` and `surface_project_coords_old` had zero call
sites (361 lines). Removed, along with `utils/raycast_torch.py`, whose only consumer
was the raycast variant. `surface_project_coords` and the two marching-cubes variants
are unchanged.

### Other uncalled code

`utils/misc.py` also held eleven uncalled helpers (`get_dd_ctd`, `get_ctd`,
`tensor_to_ctd`, `one_hot_encode_batch`, `fill_holes_3d`, …) and a `__main__` demo
block calling an undefined `surface_project_coords_sdf`; `dataloading.py` held a
duplicate heatmap-decoding chain (`heatmaps_to_coords`, `create_coordinate_tensor`,
`get_density`, `embed_patch`, `get_gt_hm*`) shadowing the one in
`verpex.geometry.heatmaps`, plus `compute_zoom`, `one_hot_encode_3d`,
`get_subreg_com`, `get_implants_poi` and `get_gruber_registration_poi`. All had zero
call sites and were removed — about 790 lines in total.

### Out-of-scope code

Dropped from the public tree (still present in the internal repository): the POI
correction/report tooling, VerSe and myeloma dataset scripts, MICCAI study scripts,
notebooks, and the personal working directories.

Two of these — `whole_scan.py` and `robsy_inference_poi.py` — were stale copies of the
inference path that still rescaled POIs *before* adding the cutout offset. That is the
bug `revert_poi_to_original_space` documents as fixed; it is silent at 1 mm isotropic
spacing and tens of millimetres off at 0.8 mm. The canonical `infer.py` path has the
fix. Do not resurrect those two files.
