import torch
import torch.nn as nn
from vertpois.geometry.surface import surface_project_coords


class SurfaceDistanceLoss(nn.Module):
    def __init__(self):
        super(SurfaceDistanceLoss, self).__init__()

    def forward(self, pred, target, mask=None, surface=None):
        if surface is None:
            return torch.tensor(0.0, device=pred.device)  # No surface provided, return zero loss

        # Compute the distance from predicted points to the surface
        dist_to_surface = surface_project_coords(pred, surface)[1]  # Get the distances to the surface
        # print("dist_to_surface", dist_to_surface.shape)
        loss = torch.norm(dist_to_surface.mean(dim=0), dim=0) / 10  # L2 distance to surface
        return loss


class WingLoss3D(nn.Module):
    def __init__(self, omega=5, epsilon=2):
        super(WingLoss3D, self).__init__()
        self.omega = torch.tensor(
            omega, dtype=torch.float32
        )  # Convert omega to a tensor
        self.epsilon = torch.tensor(
            epsilon, dtype=torch.float32
        )  # Convert epsilon to a tensor

    def forward(self, pred, target, mask=None, surface=None):
        # Compute the L1 distance between predicted and target coordinates
        delta_y = torch.abs(pred - target)

        # Compute the loss for small errors
        small_mask = delta_y < self.omega
        loss_small = self.omega * torch.log(1 + delta_y / self.epsilon) * small_mask

        # Compute the loss for large errors
        large_mask = delta_y >= self.omega
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        loss_large = (delta_y - C) * large_mask

        # Compute the total loss for all dimensions and mean over batch and points
        loss = loss_small + loss_large

        if mask is not None:
            loss = loss[mask]

        return loss.mean()


class L1LossMasked(nn.Module):
    def __init__(self):
        super(L1LossMasked, self).__init__()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None):
        loss = self.l1_loss(pred, target)
        if mask is not None:
            masked_loss = loss[mask]
        return masked_loss.mean()


class L2LossMasked(nn.Module):
    def __init__(self):
        super(L2LossMasked, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask=None, surface=None):
        loss = self.mse_loss(pred, target)
        if mask is not None:
            masked_loss = loss[mask]
        return masked_loss.mean()


class CompoundLoss(nn.Module):
    def __init__(self, loss_fns, weights=None):
        super(CompoundLoss, self).__init__()
        self.loss_fns = loss_fns
        if weights is None:
            weights = [1.0] * len(loss_fns)
        else:
            assert sum(weights) == 1.0, "Weights should sum to 1.0"
        self.weights = weights

    def forward(self, pred, target, mask=None, surface=None):
        total_loss = 0.0
        for loss_fn, weight in zip(self.loss_fns, self.weights):
            new_l = loss_fn(pred, target, mask, surface)
            # print("new_l", new_l.shape)
            total_loss += weight * new_l
        return total_loss


def get_loss_fn(loss_fn: str | list[str]):
    if isinstance(loss_fn, list):
        n = len(loss_fn)
        return CompoundLoss([get_loss_fn(lf) for lf in loss_fn], weights=[1.0 / n] * n)
    if loss_fn == "L1":
        return L1LossMasked()
    elif loss_fn == "WingLoss":
        return WingLoss3D()
    elif loss_fn == "L2":
        return L2LossMasked()
    elif loss_fn == "SD":
        return SurfaceDistanceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
