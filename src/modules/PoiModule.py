"""
Module: PoiModule

This module contains the implementation of the PoiPredictionModule class,
which is a PyTorch Lightning module for predicting points of interest (POI).
It also includes helper functions for creating feature extraction and refinement modules.

Classes:
    - PoiPredictionModule: A PyTorch Lightning module for predicting points of interest.

Functions:
    - create_feature_extraction_module: Creates a feature extraction based on a given configuration.
    - create_refinement_module: Creates a refinement module based on the given configuration.
"""

import numpy as np
import pytorch_lightning as pl
import torch

import modules.FeatureExtractionModules as feat_modules
import modules.RefinementModules as ref_modules


class PoiPredictionModule(pl.LightningModule):
    """A PyTorch Lightning module for POI prediction.

    coarse_config (dict): Configuration for the coarse feature extraction module.
        refinement_config (dict): Configuration for the refinement module.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        loss_weights (list, optional): Weights for the feature extraction and refinement losses. Defaults to None.
        optimizer (str, optional): Optimizer algorithm. Defaults to "AdamW".
        scheduler_config (dict, optional): Configuration for the learning rate scheduler. Defaults to None.
        feature_freeze_patience (int, optional): Number of epochs without improvement before freezing the feature extraction module. Defaults to None.
    Attributes:
        feature_extraction_module (Module): The feature extraction module.
        refinement_module (Module): The refinement module.
        lr (float): Learning rate for the optimizer.
        loss_weights (Tensor): Weights for the feature extraction and refinement losses.
        feature_freeze_patience (int): Number of epochs without improvement before freezing the feature extraction module.
        best_feature_loss (float): Best feature loss achieved during validation.
        val_feature_loss_outputs (list): List of feature loss values during validation.
        epochs_without_improvement (int): Number of epochs without improvement during validation.
        feature_extactor_frozen (bool): Flag indicating if the feature extraction module is frozen.
        optimizer (str): Optimizer algorithm.
        scheduler_config (dict): Configuration for the learning rate scheduler.
    Methods:
        forward(*args, **kwargs): Forward pass of the module.
        training_step(*args, **kwargs): Training step of the module.
        validation_step(*args, **kwargs): Validation step of the module.
        on_validation_epoch_end(): Callback function called at the end of each validation epoch.
        configure_optimizers(): Configures the optimizer and learning rate scheduler.
        calculate_metrics(batch, mode): Calculates metrics for the given batch and mode.
        freeze_feature_extractor(): Freezes the feature extraction module.
    """

    def __init__(
        self,
        coarse_config,
        refinement_config,
        lr=1e-4,
        loss_weights=None,
        optimizer="AdamW",
        scheduler_config=None,
        feature_freeze_patience=None,
    ):
        super().__init__()
        if loss_weights is None:
            loss_weights = [1, 1]
        self.feature_extraction_module = create_feature_extraction_module(coarse_config)
        self.refinement_module = create_refinement_module(refinement_config)
        self.lr = lr
        self.loss_weights = torch.tensor(loss_weights) / torch.sum(
            torch.tensor(loss_weights)
        )
        self.feature_freeze_patience = feature_freeze_patience
        self.best_feature_loss = np.inf
        self.val_feature_loss_outputs = []
        self.epochs_without_improvement = 0
        self.feature_extactor_frozen = False
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        """Performs the forward pass of the module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The processed batch after passing through the feature extraction and refinement modules.

        Raises:
            ValueError: If batch input is not provided.
        """

        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")

        batch = self.feature_extraction_module(batch)
        batch = self.refinement_module(batch)
        return batch

    def training_step(self, *args, **kwargs):
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        batch = self(batch)

        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = (
            feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]
        )

        metrics = self.calculate_metrics(batch, "train")
        batch_size = batch["input"].shape[0]

        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True) #Alissa: sync_dist=True (due to warning)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True) #Alissa: sync_dist=True (dur to warning)

        return loss

    def validation_step(self, *args, **kwargs):
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        batch = self(batch) #added by Alissa

        # Calculate the feature extraction loss
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        # Calculate the refinement loss
        refinement_loss = self.refinement_module.calculate_loss(batch)
        loss = (
            feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]
        )

        metrics = self.calculate_metrics(batch, "val")
        batch_size = batch["input"].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log("val_feature_loss", feature_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(
            "val_refinement_loss", refinement_loss, on_epoch=True, batch_size=batch_size, sync_dist=True
        )
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        # Check if the feature extraction module should be frozen
        if self.feature_extactor_frozen:
            return

        avg_feature_loss = torch.stack(self.val_feature_loss_outputs).mean()
        self.val_feature_loss_outputs.clear()

        if self.feature_freeze_patience is not None:
            if avg_feature_loss < self.best_feature_loss:
                self.best_feature_loss = avg_feature_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if (
                    self.epochs_without_improvement >= self.feature_freeze_patience
                    and not self.feature_extactor_frozen
                ):
                    self.freeze_feature_extractor()
                    self.feature_extactor_frozen = True
                    print("Feature extraction module frozen")

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class(self.parameters(), lr=self.lr)

        if self.scheduler_config:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, self.scheduler_config["type"]
            )
            scheduler = scheduler_class(optimizer, **self.scheduler_config["params"])

            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
            if "monitor" in self.scheduler_config:
                scheduler_config["monitor"] = self.scheduler_config["monitor"]

            return [optimizer], [scheduler_config]

        return optimizer

    def calculate_metrics(self, batch, mode):
        """Calculates metrics for the given batch and mode.

        Parameters:
            batch (Tensor): The input batch.
            mode (str): The mode of calculation.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """

        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)

        return {**feature_metrics, **refinement_metrics}

    def freeze_feature_extractor(self):
        """Freezes the feature extraction module by setting the `requires_grad`
        attribute of all its parameters to False.

        This prevents the feature extraction module from being updated during training.

        Args:
            None

        Returns:
            None
        """
        self.log("feature_frozen", True, on_epoch=True, sync_dist=True) # Alissa: added sync_dist=True
        for param in self.feature_extraction_module.parameters():
            param.requires_grad = False


def create_feature_extraction_module(config):
    """Create a feature extraction module based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        module_type: An instance of the feature extraction module.

    Raises:
        ValueError: If the provided module type is unknown.
    """

    module_type = getattr(feat_modules, config["type"])
    if module_type is None:
        raise ValueError(f"Unknown feature extraction module type: {config['type']}")

    return module_type(**config["params"])


def create_refinement_module(config):
    """Create a refinement module based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the module.

    Returns:
        object: An instance of the refinement module.

    Raises:
        ValueError: If the specified module type is unknown.

    Example:
        config = {
            "type": "SomeModule",
            "params": {
                "param1": value1,
                "param2": value2
            }
        }
        module = create_refinement_module(config)
    """

    module_type = getattr(ref_modules, config["type"])
    if module_type is None:
        raise ValueError(f"Unknown refinement module type: {config['type']}")

    return module_type(**config["params"])



class PoiNeighborPredictionModule(PoiPredictionModule):
    """
    Multi-vertebrae POI prediction module that extends PoiPredictionModule.
    
    Implements multi-task learning where the model predicts POIs for current
    vertebra and its neighbors, with different loss weights for each.
    """
    
    def __init__(
        self,
        coarse_config,
        refinement_config,
        lr=1e-4,
        loss_weights=None,
        optimizer="AdamW",
        scheduler_config=None,
        feature_freeze_patience=None,
        current_weight=1.0,
        neighbor_weight=0.2,
    ):
        """
        Args:
            current_weight (float): Loss weight for current vertebra predictions
            neighbor_weight (float): Loss weight for neighbor vertebrae predictions
            **kwargs: Arguments passed to parent PoiPredictionModule
        """
        super().__init__(
            coarse_config=coarse_config,
            refinement_config=refinement_config,
            lr=lr,
            loss_weights=loss_weights,
            optimizer=optimizer,
            scheduler_config=scheduler_config,
            feature_freeze_patience=feature_freeze_patience
        )
        
        self.current_weight = current_weight
        self.neighbor_weight = neighbor_weight
        
        # Update hyperparameters to include new parameters
        self.save_hyperparameters()

    def training_step(self, *args, **kwargs):
        """Override training step to use multi-vertebrae loss calculation"""
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        
        batch = self(batch)
        
        # Use multi-vertebrae loss calculation
        loss = self._calculate_multi_vertebrae_loss(batch)
        
        metrics = self.calculate_metrics(batch, "train")
        batch_size = batch["input"].shape[0]

        self.log("train_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, *args, **kwargs):
        """Override validation step to use multi-vertebrae loss calculation"""
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            raise ValueError("Batch input is required for the forward pass.")
        
        batch = self(batch)
        
        # Use multi-vertebrae loss calculation
        loss = self._calculate_multi_vertebrae_loss(batch)
        
        # Also calculate component losses for logging
        feature_loss = self._calculate_feature_loss_component(batch)
        refinement_loss = self._calculate_refinement_loss_component(batch)
        
        metrics = self.calculate_metrics(batch, "val")
        batch_size = batch["input"].shape[0]

        self.val_feature_loss_outputs.append(feature_loss)

        self.log("val_feature_loss", feature_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_refinement_loss", refinement_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return loss

    def _calculate_multi_vertebrae_loss(self, batch):
        """
        Calculate weighted loss for multi-vertebrae training.
        
        Args:
            batch (dict): Batch containing predictions, targets, and metadata
            
        Returns:
            torch.Tensor: Weighted total loss
        """
        # Check if this is actually a multi-vertebrae batch
        if "n_vertebrae" not in batch or "n_pois_per_vertebra" not in batch:
            # Fallback to standard loss calculation
            return self._calculate_standard_loss(batch)
        
        batch_size = batch["input"].shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            n_vertebrae = batch["n_vertebrae"] if isinstance(batch["n_vertebrae"], int) else batch["n_vertebrae"][b]
            n_pois_per_vertebra = batch["n_pois_per_vertebra"] if isinstance(batch["n_pois_per_vertebra"], int) else batch["n_pois_per_vertebra"][b]
            
            # Calculate loss for each vertebra in this batch sample
            for vert_idx in range(n_vertebrae):
                start_idx = vert_idx * n_pois_per_vertebra
                end_idx = start_idx + n_pois_per_vertebra
                
                # Extract vertebra-specific data
                vert_batch = self._extract_vertebra_batch(batch, b, start_idx, end_idx)
                
                # Calculate vertebra-specific loss
                vert_feature_loss = self.feature_extraction_module.calculate_loss(vert_batch)
                vert_refinement_loss = self.refinement_module.calculate_loss(vert_batch)
                vert_total_loss = (
                    vert_feature_loss * self.loss_weights[0] + 
                    vert_refinement_loss * self.loss_weights[1]
                )
                
                # Apply different weights: current vs neighbors
                weight = self.current_weight if vert_idx == 0 else self.neighbor_weight
                total_loss += weight * vert_total_loss
        
        # Average over batch
        return total_loss / batch_size

    def _extract_vertebra_batch(self, batch, batch_idx, start_idx, end_idx):
        """
        Extract data for a specific vertebra from the batch.
        
        Args:
            batch (dict): Full batch data
            batch_idx (int): Index in the batch dimension
            start_idx (int): Start index for POIs of this vertebra
            end_idx (int): End index for POIs of this vertebra
            
        Returns:
            dict: Batch data for specific vertebra
        """
        vert_batch = {}
        
        # Extract relevant data for this vertebra
        if "target" in batch:
            vert_batch["target"] = batch["target"][batch_idx, start_idx:end_idx].unsqueeze(0)
        
        if "loss_mask" in batch:
            vert_batch["loss_mask"] = batch["loss_mask"][batch_idx, start_idx:end_idx].unsqueeze(0)
        
        if "coarse_preds" in batch:
            vert_batch["coarse_preds"] = batch["coarse_preds"][batch_idx, start_idx:end_idx].unsqueeze(0)
            
        if "refined_preds" in batch:
            vert_batch["refined_preds"] = batch["refined_preds"][batch_idx, start_idx:end_idx].unsqueeze(0)
        
        # Include other necessary data (input, surface, etc.)
        for key in ["input", "surface"]:
            if key in batch:
                vert_batch[key] = batch[key][batch_idx].unsqueeze(0)
        
        return vert_batch

    def _calculate_feature_loss_component(self, batch):
        """Calculate feature loss component for logging purposes"""
        if "n_vertebrae" not in batch:
            return self.feature_extraction_module.calculate_loss(batch)
        
        # For multi-vertebrae, calculate weighted average of feature losses
        batch_size = batch["input"].shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            n_vertebrae = batch["n_vertebrae"] if isinstance(batch["n_vertebrae"], int) else batch["n_vertebrae"][b]
            n_pois_per_vertebra = batch["n_pois_per_vertebra"] if isinstance(batch["n_pois_per_vertebra"], int) else batch["n_pois_per_vertebra"][b]
            
            for vert_idx in range(n_vertebrae):
                start_idx = vert_idx * n_pois_per_vertebra
                end_idx = start_idx + n_pois_per_vertebra
                
                vert_batch = self._extract_vertebra_batch(batch, b, start_idx, end_idx)
                vert_loss = self.feature_extraction_module.calculate_loss(vert_batch)
                
                weight = self.current_weight if vert_idx == 0 else self.neighbor_weight
                total_loss += weight * vert_loss
        
        return total_loss / batch_size

    def _calculate_refinement_loss_component(self, batch):
        """Calculate refinement loss component for logging purposes"""
        if "n_vertebrae" not in batch:
            return self.refinement_module.calculate_loss(batch)
        
        # For multi-vertebrae, calculate weighted average of refinement losses
        batch_size = batch["input"].shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            n_vertebrae = batch["n_vertebrae"] if isinstance(batch["n_vertebrae"], int) else batch["n_vertebrae"][b]
            n_pois_per_vertebra = batch["n_pois_per_vertebra"] if isinstance(batch["n_pois_per_vertebra"], int) else batch["n_pois_per_vertebra"][b]
            
            for vert_idx in range(n_vertebrae):
                start_idx = vert_idx * n_pois_per_vertebra
                end_idx = start_idx + n_pois_per_vertebra
                
                vert_batch = self._extract_vertebra_batch(batch, b, start_idx, end_idx)
                vert_loss = self.refinement_module.calculate_loss(vert_batch)
                
                weight = self.current_weight if vert_idx == 0 else self.neighbor_weight
                total_loss += weight * vert_loss
        
        return total_loss / batch_size

    def _calculate_standard_loss(self, batch):
        """Fallback to standard loss calculation for single-vertebra batches"""
        feature_loss = self.feature_extraction_module.calculate_loss(batch)
        refinement_loss = self.refinement_module.calculate_loss(batch)
        return feature_loss * self.loss_weights[0] + refinement_loss * self.loss_weights[1]

    def calculate_metrics(self, batch, mode):
        """Override metrics calculation to include multi-vertebrae specific metrics"""
        # Get base metrics
        feature_metrics = self.feature_extraction_module.calculate_metrics(batch, mode)
        refinement_metrics = self.refinement_module.calculate_metrics(batch, mode)
        
        # Add multi-vertebrae specific metrics
        multi_metrics = self._calculate_multi_vertebrae_metrics(batch, mode)
        
        return {**feature_metrics, **refinement_metrics, **multi_metrics}

    def _calculate_multi_vertebrae_metrics(self, batch, mode):
        """Calculate vertebra-specific metrics"""
        metrics = {}
        
        if "n_vertebrae" not in batch or "coarse_preds" not in batch:
            return metrics
        
        batch_size = batch["input"].shape[0]
        n_pois_per_vertebra = batch["n_pois_per_vertebra"] if isinstance(batch["n_pois_per_vertebra"], int) else batch["n_pois_per_vertebra"][0]
        
        predictions = batch["coarse_preds"]
        targets = batch["target"]
        loss_mask = batch["loss_mask"]
        
        # Metrics for current vertebra (first n_pois_per_vertebra POIs)
        current_preds = predictions[:, :n_pois_per_vertebra]
        current_targets = targets[:, :n_pois_per_vertebra]
        current_mask = loss_mask[:, :n_pois_per_vertebra]
        
        if current_mask.any():
            current_distances = torch.norm(current_preds - current_targets, dim=-1)
            current_masked_distances = current_distances[current_mask]
            
            metrics[f"current_vertebra_mean_distance_{mode}"] = current_masked_distances.mean()
            metrics[f"current_vertebra_std_distance_{mode}"] = current_masked_distances.std()
        
        # Metrics for neighbor vertebrae (remaining POIs)
        total_pois = predictions.shape[1]
        if total_pois > n_pois_per_vertebra:
            neighbor_preds = predictions[:, n_pois_per_vertebra:]
            neighbor_targets = targets[:, n_pois_per_vertebra:]
            neighbor_mask = loss_mask[:, n_pois_per_vertebra:]
            
            if neighbor_mask.any():
                neighbor_distances = torch.norm(neighbor_preds - neighbor_targets, dim=-1)
                neighbor_masked_distances = neighbor_distances[neighbor_mask]
                
                metrics[f"neighbor_vertebrae_mean_distance_{mode}"] = neighbor_masked_distances.mean()
                metrics[f"neighbor_vertebrae_std_distance_{mode}"] = neighbor_masked_distances.std()
                
                # Ratio metrics
                if current_mask.any():
                    current_mean = current_masked_distances.mean()
                    neighbor_mean = neighbor_masked_distances.mean()
                    metrics[f"neighbor_to_current_distance_ratio_{mode}"] = neighbor_mean / current_mean
        
        return metrics

    def predict_current_vertebra_only(self, batch):
        """
        Make predictions and return only current vertebra POIs.
        Useful for inference when you only want the primary predictions.
        """
        batch = self(batch)
        
        if "n_pois_per_vertebra" not in batch:
            return batch
        
        n_pois = batch["n_pois_per_vertebra"]
        
        # Filter predictions to only current vertebra
        if "coarse_preds" in batch:
            batch["coarse_preds"] = batch["coarse_preds"][:, :n_pois]
        
        if "refined_preds" in batch:
            batch["refined_preds"] = batch["refined_preds"][:, :n_pois]
        
        if "target" in batch:
            batch["target"] = batch["target"][:, :n_pois]
            
        if "loss_mask" in batch:
            batch["loss_mask"] = batch["loss_mask"][:, :n_pois]
        
        return batch