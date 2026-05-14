import torch
import torch.nn as nn
from typing import Optional, Any
from torch_geometric.data import Data

try:
    from pygod.models import BaseDetector
except ImportError:
    # Fallback if pygod is not installed in the current environment
    class BaseDetector:
        """Dummy BaseDetector for compatibility."""
        def __init__(self, **kwargs):
            self.epoch = kwargs.get('epoch', 100)
            self.lr = kwargs.get('lr', 0.004)
            self.weight_decay = kwargs.get('weight_decay', 0.0)
            self.batch_size = kwargs.get('batch_size', 64)
            self.gpu = kwargs.get('gpu', -1)
            self.verbose = kwargs.get('verbose', 0)

from .data_adapter import DataAdapter

class DLG(BaseDetector):
    """
    Decoupled Local-to-Global Graph Neural Network (DLG)
    
    A PyGOD-compatible wrapper for the DLG model, which extracts local subgraph 
    patterns (Level 1) and global relational patterns (Level 2), and fuses them
    for robust Graph Outlier/Fraud Detection.
    """

    def __init__(self,
                 level1_dim: int = 64,
                 level2_dim: int = 64,
                 mc_dropout_rate: float = 0.1,
                 process_mode: str = "batch",
                 l1_hops: int = 2,
                 subgraph_batch_size: int = 256,
                 epoch: int = 100,
                 lr: float = 0.004,
                 weight_decay: float = 0.0,
                 batch_size: int = 64,
                 gpu: int = -1,
                 verbose: int = 0,
                 **kwargs):
        """
        Args:
            level1_dim (int): Hidden dimension for Level 1 (Local) model.
            level2_dim (int): Hidden dimension for Level 2 (Global) model.
            mc_dropout_rate (float): Dropout rate for Monte Carlo estimation.
            process_mode (str): "batch" for standard processing, "streaming" for real-time.
            l1_hops (int): Number of hops for Level 1 local subgraph extraction.
            subgraph_batch_size (int): Partition size for Level 1 subgraph extraction to save memory.
            epoch (int): Number of training epochs.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            batch_size (int): Batch size.
            gpu (int): GPU ID to use. -1 for CPU.
            verbose (int): Verbosity mode.
        """
        super(DLG, self).__init__(epoch=epoch,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  batch_size=batch_size,
                                  gpu=gpu,
                                  verbose=verbose,
                                  **kwargs)
        
        self.level1_dim = level1_dim
        self.level2_dim = level2_dim
        self.mc_dropout_rate = mc_dropout_rate
        self.process_mode = process_mode
        self.l1_hops = l1_hops
        self.subgraph_batch_size = subgraph_batch_size
        
        self.data_adapter = DataAdapter(
            mode=self.process_mode, 
            l1_hops=self.l1_hops,
            subgraph_batch_size=self.subgraph_batch_size
        )
        
        self.level1_model = None
        self.level2_model = None
        self.fusion_model = None
        
        self.device = torch.device(f'cuda:{gpu}' if gpu >= 0 and torch.cuda.is_available() else 'cpu')

    def process_graph(self, data: Data) -> Any:
        """
        Process the PyG Data object into DLG specific formats using DataAdapter.
        """
        return self.data_adapter.adapt(data)

    def _build_model(self):
        """
        Initializes the internal PyTorch modules for Level 1, Level 2, and Fusion.
        """
        self.level1_model = nn.Linear(10, self.level1_dim).to(self.device)
        self.level2_model = nn.Linear(self.level1_dim, self.level2_dim).to(self.device)
        self.fusion_model = nn.Linear(self.level1_dim + self.level2_dim, 1).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.level1_model.parameters()) + 
            list(self.level2_model.parameters()) + 
            list(self.fusion_model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def fit(self, data: Data, label: Optional[torch.Tensor] = None):
        """
        Fit the DLG model.
        """
        self._build_model()
        l1_batches, l2_data = self.process_graph(data)
        
        self.level1_model.train()
        self.level2_model.train()
        self.fusion_model.train()
        
        if self.verbose:
            print(f"Training DLG for {self.epoch} epochs on {self.device}...")
            
        for ep in range(self.epoch):
            self.optimizer.zero_grad()
            
            # Step 1: Process Level 1 Batches to get L1 embeddings
            l1_embeddings = []
            
            # Handling both List (batch) and Generator (streaming)
            for batch in l1_batches:
                batch = batch.to(self.device)
                # Dummy forward pass: in reality, pass batch.x, batch.edge_index to Level1GNN
                # Extract embeddings only for the center nodes
                out = self.level1_model(batch.x)
                center_emb = out[batch.center_mapping]
                l1_embeddings.append(center_emb)
            
            l1_full_emb = torch.cat(l1_embeddings, dim=0) # [N, level1_dim]
            
            # Step 2: Process Level 2 Graph using L1 embeddings as node features
            l2_data = l2_data.to(self.device)
            l2_out = self.level2_model(l1_full_emb) # Dummy forward
            
            # Step 3: Fusion
            score = self.fusion_model(torch.cat([l1_full_emb, l2_out], dim=-1)).squeeze(-1)
            
            # Simulated loss for demonstration
            loss = torch.mean(score) if label is None else torch.nn.functional.binary_cross_entropy_with_logits(score, label.float().to(self.device))
            loss.backward()
            self.optimizer.step()
            
            # If generator was consumed, we need to adapt again for the next epoch
            # (In streaming mode, generator is exhausted after one loop)
            if self.process_mode == "streaming" and ep < self.epoch - 1:
                l1_batches, l2_data = self.process_graph(data)
            
            if self.verbose and ep % 10 == 0:
                print(f"Epoch {ep:03d}/{self.epoch:03d} | Loss: {loss.item():.4f}")
        
        return self

    def decision_function(self, data: Data) -> torch.Tensor:
        """
        Predict raw anomaly scores.
        """
        self.level1_model.eval()
        self.level2_model.eval()
        self.fusion_model.eval()
        
        l1_batches, l2_data = self.process_graph(data)
        
        with torch.no_grad():
            l1_embeddings = []
            for batch in l1_batches:
                batch = batch.to(self.device)
                out = self.level1_model(batch.x)
                center_emb = out[batch.center_mapping]
                l1_embeddings.append(center_emb)
            
            l1_full_emb = torch.cat(l1_embeddings, dim=0)
            
            l2_data = l2_data.to(self.device)
            l2_out = self.level2_model(l1_full_emb)
            
            score = self.fusion_model(torch.cat([l1_full_emb, l2_out], dim=-1)).squeeze(-1)
            
        return score

    def predict(self, data: Data, return_confidence: bool = False, threshold: float = 0.5):
        """
        Predict binary anomalies.
        """
        score = self.decision_function(data)
        pred = (torch.sigmoid(score) > threshold).long()
        
        if return_confidence:
            return pred, torch.sigmoid(score)
        return pred
