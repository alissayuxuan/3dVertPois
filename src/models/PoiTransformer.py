import torch
import torch.nn as nn
from monai.networks.blocks.transformerblock import TransformerBlock
from typing import Optional


class PoiTransformer(nn.Module):
    def __init__(
        self,
        poi_feature_l: int,
        coord_embedding_l: int,
        poi_embedding_l: int,
        vert_embedding_l: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        n_landmarks: int,
        n_verts: int = 22,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = (
            poi_feature_l + coord_embedding_l + poi_embedding_l + vert_embedding_l
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=save_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.n_landmarks = n_landmarks 
        self.coord_embedding_l = coord_embedding_l
        self.poi_embedding_l = poi_embedding_l
        self.vert_embedding_l = vert_embedding_l

        self.coordinate_embedding = nn.Linear(3, coord_embedding_l, bias=False)

        self.poi_embedding = nn.Embedding(n_landmarks, poi_embedding_l)
        self.vert_embedding = nn.Embedding(n_verts, vert_embedding_l)

        self.norm = nn.LayerNorm(hidden_size)

        self.fine_pred = nn.Linear(hidden_size, 3)

    def forward(self, coarse_preds, poi_indices, vertebra, poi_features):
        """
        coarse_preds: (B, N_landmarks, 3)
        poi_indices: (B, N_landmarks)
        vertebra: (B)
        poi_features: (B, N_landmarks, poi_feature_l)
        """
        # Create the embeddings
        coords_embedded = self.coordinate_embedding(
            coarse_preds.float()
        )  # size (B, N_landmarks, coord_embedding_l)
        pois_embedded = self.poi_embedding(
            poi_indices
        )  # size (B, N_landmarks, poi_embedding_l)
        vert_embedded = self.vert_embedding(vertebra)  # size (B, 1, vert_embedding_l)

        # Bring vert_embedded to the same shape as the other embeddings
        vert_embedded = vert_embedded.expand(-1, self.n_landmarks, -1)

        # Concatenate the embeddings
        x = torch.cat(
            [poi_features, coords_embedded, pois_embedded, vert_embedded], dim=-1
        )  # size (B, N_landmarks, hidden_size)

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)  # size (B, N, hidden_size)
        x = self.norm(x)
        x = self.fine_pred(x)  # size (B, N, 3)

        return x
    

class FlexiblePoiTransformer(nn.Module):
    def __init__(
        self,
        poi_feature_l: Optional[int],
        coord_embedding_l: Optional[int], 
        poi_embedding_l: Optional[int],
        vert_embedding_l: Optional[int],
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        n_landmarks: int,
        n_verts: int = 22,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        # Check which components to use
        self.use_poi_features = poi_feature_l is not None and poi_feature_l > 0
        self.use_coord_embedding = coord_embedding_l is not None and coord_embedding_l > 0
        self.use_poi_embedding = poi_embedding_l is not None and poi_embedding_l > 0
        self.use_vert_embedding = vert_embedding_l is not None and vert_embedding_l > 0

        # Store dimensions (set to 0 if not used)
        self.poi_feature_l = poi_feature_l if self.use_poi_features else 0
        self.coord_embedding_l = coord_embedding_l if self.use_coord_embedding else 0
        self.poi_embedding_l = poi_embedding_l if self.use_poi_embedding else 0
        self.vert_embedding_l = vert_embedding_l if self.use_vert_embedding else 0

        # Calculate hidden size based on used components
        hidden_size = (
            self.poi_feature_l + 
            self.coord_embedding_l + 
            self.poi_embedding_l + 
            self.vert_embedding_l
        )
        
        # Ensure we have at least some input
        if hidden_size == 0:
            raise ValueError("At least one component must be enabled (poi_features, coord_embedding, poi_embedding, or vert_embedding)")

        self.hidden_size = hidden_size
        self.n_landmarks = n_landmarks

        # Only create components that are actually used
        if self.use_coord_embedding:
            self.coordinate_embedding = nn.Linear(3, coord_embedding_l, bias=False)

        if self.use_poi_embedding:
            self.poi_embedding = nn.Embedding(n_landmarks, poi_embedding_l)
            
        if self.use_vert_embedding:
            self.vert_embedding = nn.Embedding(n_verts, vert_embedding_l)

        # Core transformer components
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=save_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.fine_pred = nn.Linear(hidden_size, 3)

    def forward(self, coarse_preds=None, poi_indices=None, vertebra=None, poi_features=None):
        """
        Flexible forward pass - only processes provided inputs
        
        Args:
            coarse_preds: (B, N_landmarks, 3) - coordinates for embedding (optional)
            poi_indices: (B, N_landmarks) - POI type indices (optional)
            vertebra: (B,) - vertebra indices (optional)
            poi_features: (B, N_landmarks, poi_feature_l) - coarse features (optional)
        """
        
        # Collect feature components
        features_list = []
        
        # 1. POI Features (from coarse model)
        if self.use_poi_features:
            if poi_features is None:
                raise ValueError("poi_features is required when use_poi_features=True")
            features_list.append(poi_features)
        
        # 2. Coordinate Embeddings
        if self.use_coord_embedding:
            if coarse_preds is None:
                raise ValueError("coarse_preds is required when use_coord_embedding=True")
            coords_embedded = self.coordinate_embedding(coarse_preds.float())
            features_list.append(coords_embedded)
        
        # 3. POI Type Embeddings
        if self.use_poi_embedding:
            if poi_indices is None:
                raise ValueError("poi_indices is required when use_poi_embedding=True")
            pois_embedded = self.poi_embedding(poi_indices)
            features_list.append(pois_embedded)
        
        # 4. Vertebra Embeddings
        if self.use_vert_embedding:
            if vertebra is None:
                raise ValueError("vertebra is required when use_vert_embedding=True")
            vert_embedded = self.vert_embedding(vertebra)  # (B, 1, vert_embedding_l)
            # Expand to match landmark dimension
            vert_embedded = vert_embedded.expand(-1, self.n_landmarks, -1)
            features_list.append(vert_embedded)
        
        # Concatenate all available features
        if not features_list:
            raise ValueError("No features available - check your input arguments and enabled components")
            
        x = torch.cat(features_list, dim=-1)  # (B, N_landmarks, hidden_size)
        
        # Transformer processing
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.fine_pred(x)  # (B, N_landmarks, 3)
        
        return x

