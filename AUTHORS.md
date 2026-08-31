# Authors and provenance

This repository was extracted from an internal research codebase and published with a
fresh history, so per-commit attribution did not carry over. Contributors are recorded
here instead.

## Lineage

The pipeline originates in the master's thesis *Automated Point-of-Interest Prediction on
CT Scans of Human Vertebrae Using Spine Segmentations* by **Daniel-Jordi Regenbrecht**
(<https://github.com/doppelplusungut/3dVertPois>), and was extended in a subsequent
bachelor's thesis at the Technical University of Munich.

## Contributors

<!--
  TODO before publishing: confirm each person consents to being named here, and fill in
  what they worked on. Names below are taken from the internal repository's commit history.
-->

- Daniel-Jordi Regenbrecht — original model, dataset and training pipeline
- Hendrik Möller — VerSe support, deterministic inference, annotation-correction
  workflow, coordinate-space fixes
- (contributor) — multi-vertebra / neighbour-aware training, distributed-training fixes
- (contributor) — additional inference work

## Dependencies

Built on [TPTBox](https://github.com/Hendrik-code/TPTBox) (Apache-2.0) by Robert Graf and
Hendrik Möller. The DenseNet and sparse-DenseNet backbones are adapted from
[MONAI](https://github.com/Project-MONAI/MONAI) (Apache-2.0).
