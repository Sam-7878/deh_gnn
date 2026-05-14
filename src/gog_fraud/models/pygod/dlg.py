"""
DLG - Decoupled Local-to-Global Graph Neural Network Detector

A PyGOD-compatible detector that properly inherits from DeepDetector,
using GCN-based encoders for both local and global pattern detection.
"""

import torch
from torch_geometric.nn import GCN

from pygod.detector.base import DeepDetector
from .dlg_base import DLGBase


class DLG(DeepDetector):
    """
    Decoupled Local-to-Global Graph Neural Network (DLG)
    
    DLG is an anomaly detector that decouples graph anomaly detection into
    two levels:
    
    - **Level 1 (Local)**: A GCN encoder captures neighborhood-level
      anomalous patterns within k-hop ego-nets.
    - **Level 2 (Global)**: A GCN encoder captures graph-wide relational
      anomaly patterns using the local embeddings.
    
    The two levels are fused via a learnable gating mechanism, and anomaly
    scores are computed as weighted reconstruction errors for both node
    attributes and graph structure (similar to DOMINANT, but with the 
    Decoupled Local-to-Global architecture).
    
    Parameters
    ----------
    hid_dim : int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Split between local encoder, 
        global encoder, and decoder. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None. Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Learnable fusion weight between local and global embeddings.
        Default: ``0.5``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to reconstructed structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        The GNN backbone. Default: ``torch_geometric.nn.GCN``.
    contamination : float, optional
        Proportion of outliers in the dataset. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epochs. Default: ``100``.
    gpu : int, optional
        GPU Index, -1 for CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all. Default: ``-1``.
    weight : float, optional
        Weight between attribute and structure reconstruction error.
        Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model. Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.

    Examples
    --------
    >>> from pygod.utils import load_data
    >>> from gog_fraud.models.pygod import DLG
    >>> data = load_data("cora")
    >>> model = DLG(epoch=100)
    >>> model.fit(data)
    >>> score = model.decision_function(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 alpha=0.5,
                 sigmoid_s=False,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):
        
        super(DLG, self).__init__(
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            weight_decay=weight_decay,
            act=act,
            backbone=backbone,
            contamination=contamination,
            lr=lr,
            epoch=epoch,
            gpu=gpu,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            save_emb=save_emb,
            compile_model=compile_model,
            **kwargs
        )
        
        self.alpha = alpha
        self.weight = weight
        self.sigmoid_s = sigmoid_s

    def process_graph(self, data):
        """Compute dense adjacency for structure reconstruction."""
        DLGBase.process_graph(data)

    def init_model(self, **kwargs):
        """Initialize the DLGBase neural network."""
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return DLGBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            alpha=self.alpha,
            sigmoid_s=self.sigmoid_s,
            backbone=self.backbone,
            **kwargs
        ).to(self.device)

    def forward_model(self, data):
        """
        Forward pass: encode → decode → compute reconstruction anomaly score.
        
        This follows the same protocol as DOMINANT's forward_model,
        ensuring full compatibility with PyGOD's DeepDetector training loop.
        """
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        # Forward through DLG's Decoupled Local-to-Global architecture
        x_, s_ = self.model(x, edge_index)

        # Compute per-node anomaly score (reconstruction error)
        score = self.model.loss_func(
            x[:batch_size],
            x_[:batch_size],
            s[:batch_size, node_idx],
            s_[:batch_size],
            self.weight
        )

        loss = torch.mean(score)

        return loss, score.detach().cpu()
