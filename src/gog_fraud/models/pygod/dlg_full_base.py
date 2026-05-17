"""
DLGFullBase - Core Neural Network Module for the DLG-Full Detector

Architecture (L2 Global Encoder + Decoders):
  1. Global Encoder (GCN): Processes augmented features [original_x | L1_embeddings]
     to capture graph-wide relational patterns informed by local context.
  2. Feature Decoder (GCN): Reconstructs ORIGINAL node attributes (not augmented).
  3. Structure Decoder: Reconstructs adjacency via dot-product of embeddings.

Key Difference from DLGBase:
  - DLGBase runs local_encoder + global_encoder on the SAME graph → essentially a deeper GCN.
  - DLGFullBase's L1 is pre-trained on ego-nets (in process_graph) and FROZEN.
    This module only handles L2 global encoding on augmented features.
  - Reconstruction target is original features, not augmented ones.

Anomaly Score = α * ‖x_orig - x̂‖² + (1-α) * ‖s - ŝ‖²
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj


class DLGFullBase(nn.Module):
    """
    DLG-Full Global Encoder Module.
    
    Parameters
    ----------
    in_dim : int
        Input dimension (original_dim + L1_embedding_dim).
    orig_dim : int
        Original feature dimension (reconstruction target).
    hid_dim : int
        Hidden dimension. Default: ``64``.
    num_layers : int
        Number of GCN layers (split between encoder and decoder). Default: ``4``.
    dropout : float
        Dropout rate. Default: ``0.``.
    act : callable
        Activation function. Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool
        Whether to apply sigmoid to reconstructed structure. Default: ``False``.
    backbone : torch.nn.Module
        GNN backbone class. Default: ``GCN``.
    """
    
    def __init__(self,
                 in_dim: int,
                 orig_dim: int,
                 hid_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s: bool = False,
                 backbone=GCN,
                 **kwargs):
        super().__init__()
        
        assert num_layers >= 2, "num_layers must be >= 2"
        
        self.orig_dim = orig_dim
        encoder_layers = max(1, num_layers // 2)
        decoder_layers = max(1, num_layers - encoder_layers)
        
        # Global Encoder: processes augmented features [x_orig | L1_emb]
        self.encoder = backbone(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
            **kwargs
        )
        
        # Feature Decoder: reconstructs ORIGINAL features (orig_dim, not in_dim)
        self.attr_decoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=orig_dim,
            dropout=dropout,
            act=act,
            **kwargs
        )
        
        # Structure Decoder: dot-product
        self.sigmoid_s = sigmoid_s
        
        self.emb = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass through the L2 Global Encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Augmented node features [N, orig_dim + L1_hid_dim].
        edge_index : torch.Tensor
            Edge index [2, E].
            
        Returns
        -------
        x_ : torch.Tensor
            Reconstructed original features [N, orig_dim].
        s_ : torch.Tensor
            Reconstructed adjacency [N, N].
        """
        # Encode: augmented features → global embedding
        z = self.encoder(x, edge_index)
        
        # Save embedding
        self.emb = z.detach()
        
        # Decode: reconstruct original features
        x_ = self.attr_decoder(z, edge_index)
        
        # Decode: reconstruct structure via dot-product
        s_ = torch.mm(z, z.t())
        if self.sigmoid_s:
            s_ = torch.sigmoid(s_)
        
        return x_, s_
    
    @staticmethod
    def loss_func(x_orig, x_, s, s_, weight=0.5):
        """
        Per-node anomaly score as weighted reconstruction error.
        
        Note: x_orig is the ORIGINAL features (before L1 augmentation),
        not the augmented features. This ensures the score reflects
        how well the model can reconstruct the true node attributes
        using both local (L1) and global (L2) context.
        
        Score = weight * ‖x_orig - x̂‖² + (1-weight) * ‖s - ŝ‖²
        """
        attr_diff = torch.pow(x_orig - x_, 2)
        attr_error = torch.sqrt(torch.sum(attr_diff, dim=1))
        
        struct_diff = torch.pow(s - s_, 2)
        struct_error = torch.sqrt(torch.sum(struct_diff, dim=1))
        
        return weight * attr_error + (1 - weight) * struct_error

    @staticmethod
    def process_graph(data):
        """Compute dense adjacency matrix for structure reconstruction."""
        data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
