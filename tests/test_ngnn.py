import os
import torch
from torch_geometric.data import Data, Batch
from gog_fraud.models.level1.model import Level1ModelConfig, Level1Model
from gog_fraud.data.transforms.ngnn_online_transform import nGNNOnlineTransform

def test_ngnn_module():
    print("=== Testing Subgraph Extraction & Transformation ===")
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 16)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([1]), sample_id=99)

    transform = nGNNOnlineTransform(num_hops=1, max_nodes_per_subgraph=3)
    nested_data = transform(data)
    
    print(f"Original nodes: {data.num_nodes}")
    print(f"Extracted subgraphs: {len(nested_data.subgraph_idx.unique())}")
    assert hasattr(nested_data, 'subgraph_idx'), "subgraph_idx missing!"
    assert hasattr(nested_data, 'root_indicator'), "root_indicator missing!"
    
    print("\n=== Testing Level1nGNN Model Integration ===")
    
    cfg = Level1ModelConfig(
        in_dim=16,
        hidden_dim=32,
        out_dim=1,
        encoder_backend="ngnn",
        subgraph_pooling="mean",
        readout="mean"
    )
    
    model = Level1Model(cfg)
    
    # Simulate DataLoader batching 2 samples
    batch = Batch.from_data_list([nested_data, nested_data])
    
    out = model(batch)
    
    print(f"Output embedding shape: {out.embedding.shape}")
    print(f"Output logits shape: {out.logits.shape}")
    print(f"Output score shape: {out.score.shape}")
    
    assert out.embedding.shape[0] == 2, "Should return 2 global embeddings for 2 samples"
    
    print("\nModule successfully passed unit constraints!")

if __name__ == "__main__":
    test_ngnn_module()
