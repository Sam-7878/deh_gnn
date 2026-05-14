import torch
import pytest
from torch_geometric.data import Data
from gog_fraud.models.pygod.dlg import DLG
from gog_fraud.models.pygod.data_adapter import DataAdapter

def create_dummy_data():
    """Create a dummy PyTorch Geometric Data object for testing."""
    num_nodes = 100
    num_edges = 300
    num_features = 10
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, y=y)

def test_data_adapter_batch():
    data = create_dummy_data()
    # Use a small subgraph_batch_size to verify partition logic
    adapter = DataAdapter(mode="batch", subgraph_batch_size=32)
    l1_batches, l2 = adapter.adapt(data)
    
    assert len(l1_batches) == 4  # 100 / 32 = 4 batches (3 of 32, 1 of 4)
    assert l2 is not None
    assert l2.edge_index.size() == data.edge_index.size()

def test_data_adapter_streaming():
    data = create_dummy_data()
    adapter = DataAdapter(mode="streaming", subgraph_batch_size=32)
    l1_generator, l2 = adapter.adapt(data)
    
    batches = list(l1_generator)
    assert len(batches) == 4
    assert l2 is not None

def test_dlg_pygod_compatibility():
    """Test if DLG conforms to the standard pyGOD methods."""
    data = create_dummy_data()
    
    # Initialize the DLG wrapper with partitioning enabled
    model = DLG(epoch=2, batch_size=32, subgraph_batch_size=32, verbose=0, gpu=-1)
    
    # 1. Test fit
    model.fit(data)
    assert model.level1_model is not None
    assert model.level2_model is not None
    assert model.fusion_model is not None
    
    # 2. Test decision_function
    scores = model.decision_function(data)
    assert scores.shape[0] == data.x.size(0)
    assert isinstance(scores, torch.Tensor)
    
    # 3. Test predict
    preds = model.predict(data)
    assert preds.shape[0] == data.x.size(0)
    assert torch.all((preds == 0) | (preds == 1))
    
    preds, confs = model.predict(data, return_confidence=True)
    assert preds.shape == confs.shape
    
    print("PyGOD DLG compatibility tests passed!")

if __name__ == "__main__":
    test_data_adapter_batch()
    test_data_adapter_streaming()
    test_dlg_pygod_compatibility()
