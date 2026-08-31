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
`vertpois.geometry.spaces.revert_poi_to_original_space`.

### Unused surface-projection variants

`surface_project_coords_sdf_continuous`, `surface_project_coords_center_raycast`,
`surface_project_coords_voxel_boundary` and `surface_project_coords_old` had zero call
sites (361 lines). Removed, along with `utils/raycast_torch.py`, whose only consumer
was the raycast variant. `surface_project_coords` and the two marching-cubes variants
are unchanged.

### Out-of-scope code

Dropped from the public tree (still present on the `claude_redo` branch): the POI
correction/report tooling, VerSe and myeloma dataset scripts, MICCAI study scripts,
notebooks, and the personal working directories.

Two of these — `whole_scan.py` and `robsy_inference_poi.py` — were stale copies of the
inference path that still rescaled POIs *before* adding the cutout offset. That is the
bug `revert_poi_to_original_space` documents as fixed; it is silent at 1 mm isotropic
spacing and tens of millimetres off at 0.8 mm. The canonical `infer.py` path has the
fix. Do not resurrect those two files.
