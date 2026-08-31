"""Prediction of anatomical points-of-interest (POIs) on vertebrae.

A two-stage coarse-to-fine 3D landmark-regression pipeline: a DenseNet backbone
predicts per-landmark heatmaps on a single-vertebra cutout, and a transformer
refines those coarse coordinates using local image patches.

See the README for the training and inference workflows.
"""

__version__ = "0.1.0"
