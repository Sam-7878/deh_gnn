import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from typing import Dict, Any, Tuple, Optional, List, Generator

class DataAdapter:
    """
    Adapter class to convert standard PyTorch Geometric (PyG) `Data` objects
    into the Decoupled Local (Level 1) and Global (Level 2) representations
    required by the DLG model.
    """
    
    def __init__(self, mode: str = "batch", l1_hops: int = 2, subgraph_batch_size: int = 256):
        """
        Args:
            mode (str): "batch" or "streaming".
            l1_hops (int): Number of hops for Level 1 local subgraph extraction.
            subgraph_batch_size (int): Size of the partition to prevent O(N^2) memory issues.
        """
        self.mode = mode
        self.l1_hops = l1_hops
        self.subgraph_batch_size = subgraph_batch_size

    def adapt(self, data: Data) -> Tuple[Any, Any]:
        """
        Converts a standard PyG `Data` into (Level 1 Data, Level 2 Data).
        
        Args:
            data (Data): Standard PyG Data object.
            
        Returns:
            Tuple containing:
                - Level 1 Data (List of Batched subgraphs, or a Generator)
                - Level 2 Data (Global Graph retaining original topology)
        """
        level2_data = self._build_level2(data)
        
        if self.mode == "batch":
            level1_data = self._adapt_batch(data)
            return level1_data, level2_data
        elif self.mode == "streaming":
            level1_data = self._adapt_streaming(data)
            return level1_data, level2_data
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'batch' or 'streaming'.")

    def _build_level2(self, data: Data) -> Data:
        """
        Returns the Level 2 Global Graph.
        To maintain natural topology and ensure compatibility with pyGOD benchmarks,
        we retain the original edge_index. 
        Node features will be replaced by L1 outputs during the forward pass.
        """
        return Data(
            x=data.x, 
            edge_index=data.edge_index, 
            y=data.y if hasattr(data, 'y') else None
        )

    def _extract_ego_nets(self, data: Data, node_indices: torch.Tensor) -> List[Data]:
        """
        Extracts k-hop ego-nets for the given node indices.
        """
        subgraphs = []
        for node_idx in node_indices:
            node_idx_item = int(node_idx.item())
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx_item, self.l1_hops, data.edge_index, relabel_nodes=True, 
                num_nodes=data.x.size(0)
            )
            
            sub_x = data.x[subset]
            sub_y = data.y[node_idx_item].view(1) if hasattr(data, 'y') and data.y is not None else None
            
            subgraph = Data(
                x=sub_x, 
                edge_index=sub_edge_index, 
                y=sub_y,
                center_node_idx=torch.tensor([node_idx_item], dtype=torch.long),
                # Mapping tells us which node in the subgraph is the center node
                center_mapping=mapping.view(1)
            )
            subgraphs.append(subgraph)
        return subgraphs

    def _adapt_batch(self, data: Data) -> List[Batch]:
        """
        Batch extraction logic.
        Extracts local ego-graphs for all nodes and batches them in chunks of `subgraph_batch_size`.
        This partition-based approach reduces memory from O(N^2) to O(K^2) per batch.
        """
        num_nodes = data.x.size(0)
        all_nodes = torch.arange(num_nodes)
        
        batched_subgraphs = []
        for i in range(0, num_nodes, self.subgraph_batch_size):
            batch_nodes = all_nodes[i:i + self.subgraph_batch_size]
            subgraphs = self._extract_ego_nets(data, batch_nodes)
            batch = Batch.from_data_list(subgraphs)
            batched_subgraphs.append(batch)
            
        return batched_subgraphs

    def _adapt_streaming(self, data: Data) -> Generator[Batch, None, None]:
        """
        Streaming extraction logic.
        Yields batches of subgraphs one by one to save memory during real-time inference.
        """
        num_nodes = data.x.size(0)
        all_nodes = torch.arange(num_nodes)
        
        for i in range(0, num_nodes, self.subgraph_batch_size):
            batch_nodes = all_nodes[i:i + self.subgraph_batch_size]
            subgraphs = self._extract_ego_nets(data, batch_nodes)
            yield Batch.from_data_list(subgraphs)
