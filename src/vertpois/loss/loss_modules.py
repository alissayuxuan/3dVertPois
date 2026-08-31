"""Loss functions for landmark regression.

All losses share the signature ``(pred, target, mask=None, surface=None)`` so they
are interchangeable in a config, and so :class:`CompoundLoss` can combine any of
them. ``pred`` and ``target`` are ``(batch, n_landmarks, 3)`` coordinate tensors;
``mask`` is a ``(batch, n_landmarks)`` boolean selecting the landmarks that count.

Callers pass coordinates already scaled to millimetres, so a loss value means the
same thing regardless of the voxel spacing the sample was acquired at.
"""

import torch
from torch import nn

from vertpois.geometry.surface import surface_project_coords


class SurfaceDistanceLoss(nn.Module):
    """Penalise predictions that lie off the vertebra surface.

    Ignores ``target`` entirely: it measures only how far each prediction sits from
    the surface mesh, which is useful as one term of a :class:`CompoundLoss`
    alongside a term that does compare against the ground truth.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None, surface=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean distance from the predictions to the surface.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Unused; present so every loss shares one signature.
            mask: Unused by this loss.
            surface: Surface point cloud to project onto. When ``None`` the loss is
                zero, so a config can enable this term only where surfaces exist.

        Returns:
            A scalar loss tensor.
        """
        if surface is None:
            return torch.tensor(0.0, device=pred.device)

        dist_to_surface = surface_project_coords(pred, surface)[1]
        # Scaled down by 10 to keep this term commensurate with the coordinate
        # losses it is usually compounded with.
        return torch.norm(dist_to_surface.mean(dim=0), dim=0) / 10


class WingLoss3D(nn.Module):
    """Wing loss, which weights small landmark errors more heavily than L1.

    Logarithmic inside ``omega`` and linear outside it, so sub-``omega`` refinement
    keeps producing gradient instead of flattening out the way L1 does.

    Args:
        omega: Error magnitude at which the loss switches from log to linear.
        epsilon: Curvature of the logarithmic part; smaller is steeper.
    """

    def __init__(self, omega=5, epsilon=2):
        super().__init__()
        self.omega = torch.tensor(omega, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)

    def forward(self, pred, target, mask=None, surface=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean wing loss over the selected landmarks.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)`` selecting landmarks to score.
            surface: Unused; present so every loss shares one signature.

        Returns:
            A scalar loss tensor.
        """
        delta_y = torch.abs(pred - target)

        small_mask = delta_y < self.omega
        loss_small = self.omega * torch.log(1 + delta_y / self.epsilon) * small_mask

        # Offset chosen so the linear part meets the logarithmic part at omega.
        large_mask = delta_y >= self.omega
        c = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        loss_large = (delta_y - c) * large_mask

        loss = loss_small + loss_large
        if mask is not None:
            loss = loss[mask]
        return loss.mean()


class L1LossMasked(nn.Module):
    """Mean absolute coordinate error over the landmarks the mask selects."""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean L1 error.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)``. When ``None`` every landmark counts.
            surface: Unused; present so every loss shares one signature.

        Returns:
            A scalar loss tensor.
        """
        loss = self.l1_loss(pred, target)
        if mask is not None:
            loss = loss[mask]
        return loss.mean()


class L2LossMasked(nn.Module):
    """Mean squared coordinate error over the landmarks the mask selects."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None) -> torch.Tensor:  # noqa: ARG002
        """Return the mean squared error.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)``. When ``None`` every landmark counts.
            surface: Unused; present so every loss shares one signature.

        Returns:
            A scalar loss tensor.
        """
        loss = self.mse_loss(pred, target)
        if mask is not None:
            loss = loss[mask]
        return loss.mean()


class CompoundLoss(nn.Module):
    """Weighted sum of several losses.

    Args:
        loss_fns: The losses to combine.
        weights: One weight per loss, summing to 1. Defaults to equal weights.

    Raises:
        AssertionError: If explicit weights do not sum to 1.
    """

    def __init__(self, loss_fns, weights=None):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            weights = [1.0] * len(loss_fns)
        else:
            assert sum(weights) == 1.0, "Weights should sum to 1.0"
        self.weights = weights

    def forward(self, pred, target, mask=None, surface=None) -> torch.Tensor:
        """Return the weighted sum of the component losses.

        Args:
            pred: Predicted coordinates, ``(batch, n_landmarks, 3)``.
            target: Ground-truth coordinates, same shape.
            mask: Boolean ``(batch, n_landmarks)`` passed to each component.
            surface: Surface point cloud passed to each component.

        Returns:
            A scalar loss tensor.
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            total_loss += weight * loss_fn(pred, target, mask, surface)
        return total_loss


def get_loss_fn(loss_fn: str | list[str]) -> nn.Module:
    """Resolve a loss by name, as written in an experiment config.

    Args:
        loss_fn: One of ``"L1"``, ``"L2"``, ``"WingLoss"`` or ``"SD"``, or a list of
            them, which builds an equally weighted :class:`CompoundLoss`.

    Returns:
        The loss module.

    Raises:
        ValueError: If the name is not recognised.
    """
    if isinstance(loss_fn, list):
        n = len(loss_fn)
        return CompoundLoss([get_loss_fn(lf) for lf in loss_fn], weights=[1.0 / n] * n)
    if loss_fn == "L1":
        return L1LossMasked()
    if loss_fn == "WingLoss":
        return WingLoss3D()
    if loss_fn == "L2":
        return L2LossMasked()
    if loss_fn == "SD":
        return SurfaceDistanceLoss()
    raise ValueError(f"Unknown loss function: {loss_fn}")
