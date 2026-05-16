"""
DLGBase - Core Neural Network Module for the DLG (Decoupled Local-to-Global) Detector

Architecture:
  1. Local Encoder (Level 1): GCN over k-hop ego-net subgraphs → local node embedding
  2. Global Encoder (Level 2): GCN over the full graph using L1 embeddings → global node embedding
  3. Feature Decoder: Reconstructs original node attributes from the fused embedding
  4. Structure Decoder: Reconstructs adjacency via dot-product of embeddings

Anomaly Score = α * ‖x - x̂‖² + (1-α) * ‖s - ŝ‖²  (per-node reconstruction error)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj


class DotProductDecoder(nn.Module):
    """Reconstructs adjacency matrix via dot product of node embeddings."""
    
    def __init__(self, sigmoid: bool = False):
        super().__init__()
        self.sigmoid = sigmoid
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s_ = torch.mm(z, z.t())
        if self.sigmoid:
            s_ = torch.sigmoid(s_)
        return s_


class DLGBase(nn.Module):
    """
    Decoupled Local-to-Global Graph Neural Network (Base Module)
    
    This module implements the core DLG architecture:
    - A Local GCN encoder that captures neighborhood-level patterns
    - A Global GCN encoder that captures graph-wide relational patterns
    - Reconstruction decoders for both attributes and structure
    
    The decoupled design allows DLG to separately model local anomalies
    (unusual node attributes within their neighborhood) and global anomalies
    (unusual structural patterns at the graph level).
    
    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension for both encoders. Default: ``64``.
    num_layers : int
        Total number of GCN layers (split between local and global). Default: ``4``.
    dropout : float
        Dropout rate. Default: ``0.``.
    act : callable
        Activation function. Default: ``torch.nn.functional.relu``.
    alpha : float
        Weight balancing local vs global embeddings in fusion. Default: ``0.5``.
    sigmoid_s : bool
        Whether to apply sigmoid to reconstructed structure. Default: ``False``.
    backbone : torch.nn.Module
        GNN backbone class. Default: ``GCN``.
    """
    
    def __init__(self,
                 in_dim: int,
                 hid_dim: int = 64,
                 num_layers: int = 4,
                 dropout: float = 0.,
                 act=torch.nn.functional.relu,
                 alpha: float = 0.5,
                 sigmoid_s: bool = False,
                 backbone=GCN,
                 **kwargs):
        super().__init__()
        
        assert num_layers >= 2, "num_layers must be >= 2"
        
        # Split layers between local and global encoders
        local_layers = max(1, num_layers // 2)
        global_layers = max(1, num_layers - local_layers)
        
        # Level 1: Local Encoder — captures neighborhood-level patterns
        self.local_encoder = backbone(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=local_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
            **kwargs
        )
        
        # Level 2: Global Encoder — captures graph-wide relational patterns
        self.global_encoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=global_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
            **kwargs
        )
        
        # Fusion gate: learnable weight to balance local vs global
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        # Feature Decoder: reconstructs original node attributes
        decoder_layers = max(1, num_layers // 2)
        self.attr_decoder = backbone(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=in_dim,
            dropout=dropout,
            act=act,
            **kwargs
        )
        
        # Structure Decoder: reconstructs adjacency via dot product
        self.struct_decoder = DotProductDecoder(sigmoid=sigmoid_s)
        
        self.emb = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass through the Decoupled Local-to-Global architecture.
        
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [N, in_dim].
        edge_index : torch.Tensor
            Edge index [2, E].
            
        Returns
        -------
        x_ : torch.Tensor
            Reconstructed node features [N, in_dim].
        s_ : torch.Tensor
            Reconstructed adjacency matrix [N, N].
        """
        # Level 1: Local encoding
        z_local = self.local_encoder(x, edge_index)
        
        # Level 2: Global encoding (using local embeddings as input)
        z_global = self.global_encoder(z_local, edge_index)
        
        # Fusion: learnable combination of local and global embeddings
        alpha = torch.sigmoid(self.alpha)
        z_fused = alpha * z_local + (1 - alpha) * z_global
        
        # Save embedding for potential downstream use
        self.emb = z_fused.detach()
        
        # Decode: reconstruct features
        x_ = self.attr_decoder(z_fused, edge_index)
        
        # Decode: reconstruct structure
        s_ = self.struct_decoder(z_fused)
        
        return x_, s_
    
    @staticmethod
    def loss_func(x, x_, s, s_, weight=0.5):
        """
        Compute per-node anomaly score as weighted reconstruction error.
        
        Score = weight * ‖x - x̂‖² + (1-weight) * ‖s - ŝ‖²
        
        Parameters
        ----------
        x : torch.Tensor
            Original node features [N, D].
        x_ : torch.Tensor
            Reconstructed features [N, D].
        s : torch.Tensor
            Original adjacency [N, N'] (may be subsampled).
        s_ : torch.Tensor
            Reconstructed adjacency [N, N'].
        weight : float
            Balance between attribute and structure error.
            
        Returns
        -------
        score : torch.Tensor
            Per-node anomaly scores [N].
        """
        # Attribute reconstruction error
        attr_diff = torch.pow(x - x_, 2)
        attr_error = torch.sqrt(torch.sum(attr_diff, dim=1))
        
        # Structure reconstruction error
        struct_diff = torch.pow(s - s_, 2)
        struct_error = torch.sqrt(torch.sum(struct_diff, dim=1))
        
        score = weight * attr_error + (1 - weight) * struct_error
        return score

    @staticmethod
    def process_graph(data):
        """Compute dense adjacency matrix for structure reconstruction."""
        data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
