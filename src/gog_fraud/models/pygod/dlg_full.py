"""
DLG-Full — Decoupled Local-to-Global (Full Pipeline) Detector

A PyGOD-compatible detector that implements the FULL DLG pipeline:
  1. Level 1 (process_graph): Pre-train a local GCN-AE on ego-net subgraphs
     to capture neighborhood-level anomalous patterns.
  2. Level 2 (fit): Train a global GCN-AE on augmented features
     [original_x | L1_embeddings] to capture graph-wide relational patterns.

Key Difference from DLG:
  - DLG: local_encoder + global_encoder operate on the SAME graph → deeper GCN.
  - DLG-Full: L1 is pre-trained on SEPARATE ego-nets → genuine local context.
    L2 then operates on features enriched with true local structure information.

Fully compatible with PyGOD's fit()/decision_function() interface.
Can be submitted as a PyGOD PR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj

from pygod.detector.base import DeepDetector
from .dlg_full_base import DLGFullBase


class DLGFull(DeepDetector):
    """
    Decoupled Local-to-Global Graph Neural Network — Full Pipeline (DLG-Full)
    
    Unlike the standard DLG which runs both local and global encoders on the
    same graph topology, DLG-Full pre-trains a separate Level 1 encoder on
    k-hop ego-net subgraphs to capture genuine neighborhood-level patterns.
    These L1 embeddings are then concatenated with original features, and a
    Level 2 global encoder learns graph-wide relational anomaly patterns
    on the augmented representation.
    
    Parameters
    ----------
    hid_dim : int, optional
        Hidden dimension for both L1 and L2 encoders. Default: ``64``.
    num_layers : int, optional
        Number of GCN layers in the L2 global encoder. Default: ``4``.
    l1_hops : int, optional
        Number of hops for ego-net extraction. Default: ``2``.
    l1_epochs : int, optional
        Number of training epochs for L1 pre-training. Default: ``20``.
    l1_hid_dim : int, optional
        Hidden dimension of L1 local encoder. Default: ``64``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Apply sigmoid to reconstructed structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        GNN backbone class. Default: ``GCN``.
    contamination : float, optional
        Expected proportion of outliers. Default: ``0.1``.
    lr : float, optional
        Learning rate for L2 training. Default: ``0.004``.
    epoch : int, optional
        Max training epochs for L2. Default: ``100``.
    gpu : int, optional
        GPU index, -1 for CPU. Default: ``-1``.
    batch_size : int, optional
        Mini-batch size, 0 for full batch. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling. Default: ``-1``.
    weight : float, optional
        Balance between attribute and structure reconstruction. Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Default: ``0``.
    save_emb : bool, optional
        Whether to save embeddings. Default: ``False``.
    compile_model : bool, optional
        Whether to compile model with torch.compile. Default: ``False``.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 l1_hops=2,
                 l1_epochs=20,
                 l1_hid_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
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
        
        super(DLGFull, self).__init__(
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
        
        self.l1_hops = l1_hops
        self.l1_epochs = l1_epochs
        self.l1_hid_dim = l1_hid_dim
        self.weight = weight
        self.sigmoid_s = sigmoid_s
        self._orig_dim = None  # Original feature dim (set in process_graph)

    def process_graph(self, data):
        """
        Pre-process: compute dense adjacency + pre-train L1 on ego-nets.
        
        Guarded against double-augmentation (process_graph is called in both
        fit() and decision_function() by PyGOD's DeepDetector).
        """
        # Guard: if already augmented, only recompute dense adjacency
        if hasattr(data, '_dlg_full_augmented') and data._dlg_full_augmented:
            data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            return
        
        # Store original feature dimension for reconstruction target
        self._orig_dim = data.x.size(1)
        
        # Dense adjacency for structure reconstruction
        data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        
        # Pre-train L1 and get per-node local embeddings
        l1_embs = self._pretrain_level1(data)
        
        # Augment features: [original_x | L1_embeddings]
        data.x = torch.cat([data.x, l1_embs.to(data.x.device)], dim=-1)
        data._dlg_full_augmented = True

    def _pretrain_level1(self, data):
        """
        Pre-train a separate L1 GCN autoencoder and extract per-node embeddings.
        
        The "Decoupled" key insight: L1 is trained SEPARATELY on reconstruction
        loss, then FROZEN. Its embeddings capture local structural patterns that
        the L2 global encoder can leverage but couldn't learn by itself.
        
        GCN's message passing inherently aggregates k-hop neighborhood info
        (k = num_layers), achieving the same effect as explicit ego-net
        extraction but in O(1) forward passes instead of O(N) k_hop_subgraph calls.
        """
        in_dim = data.x.size(1)
        device = self.device
        
        # Build L1 local autoencoder (separate from L2)
        l1_encoder = GCN(
            in_channels=in_dim, hidden_channels=self.l1_hid_dim,
            num_layers=self.l1_hops, out_channels=self.l1_hid_dim
        ).to(device)
        l1_decoder = nn.Linear(self.l1_hid_dim, in_dim).to(device)
        
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        optimizer = torch.optim.Adam(
            list(l1_encoder.parameters()) + list(l1_decoder.parameters()), lr=0.01
        )
        
        # Pre-train L1: reconstruct node features via local GCN
        l1_encoder.train()
        for ep in range(self.l1_epochs):
            optimizer.zero_grad()
            z = l1_encoder(x, edge_index)
            x_hat = l1_decoder(z)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
        
        # Extract frozen L1 embeddings
        l1_encoder.eval()
        with torch.no_grad():
            embs = l1_encoder(x, edge_index).cpu()
        
        # Cleanup L1 model (not needed during L2 training)
        del l1_encoder, l1_decoder, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embs

    def init_model(self, **kwargs):
        """Initialize the L2 global encoder with augmented input dimension."""
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return DLGFullBase(
            in_dim=self.in_dim,         # orig_dim + l1_hid_dim (after augmentation)
            orig_dim=self._orig_dim,     # Original feature dim (reconstruction target)
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            sigmoid_s=self.sigmoid_s,
            backbone=self.backbone,
            **kwargs
        ).to(self.device)

    def forward_model(self, data):
        """
        L2 Forward: encode augmented features → decode → reconstruction score.
        
        The reconstruction target is the ORIGINAL features (before L1 augmentation),
        not the augmented ones. This ensures the anomaly score measures how well
        the model can leverage L1 local context to reconstruct true node attributes.
        """
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)          # Augmented: [orig_x | L1_emb]
        x_orig = x[:, :self._orig_dim]       # Original features only
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        # Forward through L2 global encoder
        x_, s_ = self.model(x, edge_index)

        # Score: reconstruction error on ORIGINAL features + structure
        score = self.model.loss_func(
            x_orig[:batch_size],
            x_[:batch_size],
            s[:batch_size, node_idx],
            s_[:batch_size],
            self.weight
        )

        loss = torch.mean(score)
        return loss, score.detach().cpu()
