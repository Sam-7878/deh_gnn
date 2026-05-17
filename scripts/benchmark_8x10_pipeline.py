"""
Extended Benchmark Pipeline for DLG SCI Paper
==============================================
Compares 8 models across up to 10 diverse-domain datasets.

Models (6 strong PyGOD baselines + 2 DLG variants):
  - DOMINANT    — GCN Autoencoder (foundational baseline)
  - AnomalyDAE  — Dual Autoencoder (attribute + structure)
  - CoLA        — Contrastive subgraph-level detection
  - CONAD       — Contrastive + Data Augmentation (DOMINANT enhanced)
  - GADNR       — Neighborhood Reconstruction (2024 SOTA)
  - OCGNN       — One-Class GCN (SVM-style boundary)
  - DLG-Base    — Decoupled Local-to-Global (L2 only, no L1 pre-training)
  - DLG         — Full Decoupled Local-to-Global (L1→L2 pipeline, ours)

Datasets (diverse domains, increasing scale):
  ── Fraud / AML / Financial ──────────────────────────────
  1. Elliptic     (203,769 nodes)  — Bitcoin AML (licit vs illicit)
  2. DGraphFin    (3,700,550 nodes) — Large-scale Financial Loan Fraud
  3. Yelp         (716,847 nodes)  — Review Spam / Fraud
  4. Amazon       (11,944 nodes)   — E-commerce Review Fraud
  ── Blockchain / Trust ───────────────────────────────────
  5. BitcoinOTC   (5,881 nodes)    — Bitcoin Trust Network
  ── Social Network Anomaly ───────────────────────────────
  6. Flickr       (89,250 nodes)   — Spam Account Detection
  7. Reddit       (232,965 nodes)  — Troll / Sybil Detection
  ── Citation (Sanity Check + Baseline) ───────────────────
  8. Cora         (2,708 nodes)    — Small-scale sanity check
  9. CiteSeer     (3,327 nodes)    — Cross-domain validation
  10. PubMed      (19,717 nodes)   — Medium-scale medical

All datasets are stored in: /mnt/d/_Work/_data/DLG/<dataset_name>/
"""

import os
import sys
import time
import traceback
import psutil
import gc

import torch
import pandas as pd
from torch_geometric.datasets import (
    Planetoid, Flickr, Reddit,
    EllipticBitcoinDataset, Yelp,
    Amazon, BitcoinOTC,
)
from pygod.generator import gen_contextual_outlier, gen_structural_outlier

# PyGOD Baselines
## GUIDE is not supported yet.
## GUIDE model is too time consuming.
# from pygod.detector import DOMINANT, AnomalyDAE, CoLA, CONAD, GUIDE, OCGNN
from pygod.detector import DOMINANT, AnomalyDAE, CoLA, CONAD, OCGNN

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from gog_fraud.models.pygod.dlg import DLG as DLGBase
from gog_fraud.models.pygod.dlg_full import DLGFull as DLG
from gog_fraud.models.pygod.gadnr import GADNR

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Configuration
# ==========================================
DATA_ROOT = "/mnt/d/_Work/_data/DLG"
REPORT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../docs/work_reports/25-dlg_full_pipeline_benchmark'
))

# Partition size for dense-adj models (DOMINANT, CONAD, DLG).
# 16384² × 4 bytes = ~1GB per partition → perfectly utilizes VRAM while staying safe.
PARTITION_SIZE = 16384
DGraphFin_PARTITION_SIZE = 4096
Yelp_PARTITION_SIZE = 4096
Reddit_PARTITION_SIZE = 8192

# Subsample limit removed. With GPU acceleration and graph partitioning,
# even massive datasets like DGraphFin (3.7M) and Yelp (716K) can be fully processed.
MAX_LARGE_SUBSAMPLE = None


# ==========================================
# 1. Dataset Loaders
# ==========================================

def _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03, m_clique=10, k=50):
    """Inject synthetic contextual and structural outliers into a PyG Data object."""
    n_contextual = max(10, int(data.num_nodes * contextual_ratio))
    data, yc = gen_contextual_outlier(data, n=n_contextual, k=k, seed=42)

    n_clique = max(1, int((data.num_nodes * structural_ratio) / m_clique))
    data, ys = gen_structural_outlier(data, m=m_clique, n=n_clique, seed=42)

    data.y = torch.logical_or(yc, ys).long()
    return data


def _repackage_graph(data):
    """Repackage graph with self-loops, edge validation, and feature sanitization.
    Adapted from legacy_adapter._repackage_minimal — proven on Ethereum graphs.
    """
    from torch_geometric.utils import add_self_loops, coalesce
    from torch_geometric.data import Data as PyGData

    num_nodes = data.num_nodes
    edge_index = data.edge_index.long()

    # Remove out-of-range edges
    if edge_index.numel() > 0:
        valid = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) \
              & (edge_index[0] >= 0) & (edge_index[1] >= 0)
        if not valid.all():
            edge_index = edge_index[:, valid]

    # Add self-loops so every node is represented in edge_index
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index)

    # Sanitize features: replace NaN/Inf with 0 (critical for Yelp)
    x = data.x.float()
    nan_count = torch.isnan(x).sum().item() + torch.isinf(x).sum().item()
    if nan_count > 0:
        print(f"    ⚠ Sanitized {nan_count:,} NaN/Inf values in features")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    clean = PyGData(x=x, edge_index=edge_index, num_nodes=num_nodes)
    if hasattr(data, 'y') and data.y is not None:
        clean.y = data.y
    return clean


def _maybe_subsample(data, name, max_nodes):
    """Downsample using multi-seed BFS, then repackage for PyGOD safety."""
    if not max_nodes or data.num_nodes <= max_nodes:
        return _repackage_graph(data)

    print(f"    ↳ Subsampling {name}: {data.num_nodes:,} → {max_nodes:,} nodes")
    from torch_geometric.utils import k_hop_subgraph

    collected = torch.tensor([], dtype=torch.long)
    tried_seeds = set()

    for _ in range(50):
        seed = torch.randint(0, data.num_nodes, (1,)).item()
        if seed in tried_seeds:
            continue
        tried_seeds.add(seed)
        subset, _, _, _ = k_hop_subgraph(
            seed, num_hops=2, edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        collected = torch.unique(torch.cat([collected, subset]))
        if len(collected) >= max_nodes:
            collected = collected[:max_nodes]
            break

    if len(collected) < max_nodes:
        all_idx = torch.arange(data.num_nodes)
        mask_taken = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask_taken[collected] = True
        extra = all_idx[~mask_taken][torch.randperm((~mask_taken).sum())[:max_nodes - len(collected)]]
        collected = torch.cat([collected, extra])

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[collected] = True
    data = data.subgraph(mask)

    data = _repackage_graph(data)
    avg_degree = data.num_edges / max(data.num_nodes, 1)
    print(f"    ↳ Result: {data.num_nodes:,} nodes, {data.num_edges:,} edges (avg deg={avg_degree:.1f})")
    return data


# ── Fraud / AML / Financial ────────────────────────────────

def load_elliptic():
    """Elliptic Bitcoin: 203K nodes, real AML labels (licit/illicit/unknown)."""
    root = os.path.join(DATA_ROOT, "Elliptic")
    print("  [Dataset] Elliptic Bitcoin (203K nodes, AML)...")
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)
    
    # Remove unknown-label nodes — use ALL known nodes (no subsample)
    known_mask = (data.y != 2)
    data = data.subgraph(known_mask)
    print(f"    ↳ After removing unknown labels: {data.num_nodes:,} nodes (full)")
    data = _repackage_graph(data)
    return data


def load_dgraphfin():
    """DGraphFin: 3.7M nodes, large-scale financial loan fraud network.
    Loads directly from the extracted dgraphfin.npz file.
    Labels: 0=normal, 1=fraud, 2/3=background(unknown).
    """
    npz_path = os.path.join(DATA_ROOT, "DGraphFin", "dgraphfin.npz")
    
    if not os.path.exists(npz_path):
        print(f"  [SKIP] DGraphFin: File not found at {npz_path}")
        return None
    
    import numpy as np
    print("  [Dataset] DGraphFin (3.7M nodes, Financial Fraud)...")
    print("    ↳ Loading dgraphfin.npz (680MB, this may take a moment)...")
    
    with np.load(npz_path) as loader:
        x = torch.from_numpy(loader['x']).float()
        y = torch.from_numpy(loader['y']).long()
        edge_index = torch.from_numpy(loader['edge_index']).long().t().contiguous()
    
    if y.dim() > 1:
        y = y.squeeze(-1)
        
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"    ↳ Full graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    
    # Keep only nodes with known labels (0=normal, 1=fraud), drop background (2,3)
    known_mask = (data.y == 0) | (data.y == 1)
    data = data.subgraph(known_mask)
    print(f"    ↳ After removing background labels: {data.num_nodes:,} nodes")
    
    # DGraphFin is very large — subsample to 100K for practical runtime
    data = _maybe_subsample(data, "DGraphFin", MAX_LARGE_SUBSAMPLE)
    return data


def load_yelp():
    """Yelp: 716K nodes, customer review network (spam/fraud review detection)."""
    root = os.path.join(DATA_ROOT, "Yelp")
    print("  [Dataset] Yelp (716K nodes, Review Fraud)...")
    dataset = Yelp(root=root)
    data = dataset[0]
    
    # Yelp has multi-label y. Convert to binary anomaly: any positive label = anomaly.
    if data.y.dim() > 1:
        data.y = (data.y.sum(dim=-1) > 0).long()
    
    # Yelp is large — subsample to 100K for practical runtime
    data = _maybe_subsample(data, "Yelp", MAX_LARGE_SUBSAMPLE)
    
    # Inject structural outliers to create graph-level anomalies
    data = _inject_outliers(data, contextual_ratio=0.01, structural_ratio=0.01, m_clique=8)
    return data


def load_amazon():
    """Amazon: ~12K nodes, e-commerce review fraud detection.
    Multi-relational graph (users connected by shared product/rating/text).
    """
    root = os.path.join(DATA_ROOT, "Amazon")
    print("  [Dataset] Amazon (12K nodes, E-commerce Review Fraud)...")
    dataset = Amazon(root=root, name='Computers')
    data = dataset[0]
    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.02, m_clique=8)
    return data


# ── Blockchain / Trust ──────────────────────────────────────

def load_bitcoin_otc():
    """Bitcoin-OTC: 5.8K nodes, who-trusts-whom Bitcoin trading network.
    Uses the last temporal snapshot (most mature trust network).
    Edge weights represent trust ratings (-10 to +10).
    """
    root = os.path.join(DATA_ROOT, "BitcoinOTC")
    print("  [Dataset] Bitcoin-OTC (5.8K nodes, Trust Network)...")
    dataset = BitcoinOTC(root=root, edge_window_size=10)
    # Use last snapshot (most complete graph)
    data = dataset[-1]
    
    # BitcoinOTC has no node features → generate degree-based features
    from torch_geometric.utils import degree
    num_nodes = data.num_nodes
    deg = degree(data.edge_index[0], num_nodes=num_nodes)
    # Create simple node features: [degree, log_degree, in/out ratio, ...]
    log_deg = torch.log1p(deg)
    # Stack a few structural features
    x = torch.stack([deg, log_deg, deg / (deg.max() + 1e-6)], dim=-1).float()
    data.x = x
    data.y = torch.zeros(num_nodes, dtype=torch.long)  # will be overwritten by outlier injection
    
    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03, m_clique=8)
    return data


# ── Social Network Anomaly ──────────────────────────────────

def load_twitch():
    """Twitch-EN: 7.1K nodes, Bot/Sybil Detection.
    Note: The upstream server for this dataset (graphmining.ai) frequently
    returns HTTP 404 or times out. We catch this to prevent pipeline crashes.
    """
    from torch_geometric.datasets import Twitch
    root = os.path.join(DATA_ROOT, "Twitch")
    print("  [Dataset] Twitch-EN (7.1K nodes, Bot/Sybil Detection)...")
    try:
        dataset = Twitch(root=root, name="EN")
        data = dataset[0]
        data = _repackage_graph(data)
        data = _inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.02, m_clique=8)
        return data
    except Exception as e:
        print(f"  [SKIP] Twitch-EN: Failed to download/load dataset. Upstream server may be down. ({e})")
        return None


def load_flickr():
    """Flickr: 89K nodes, image/social network for spam detection."""
    root = os.path.join(DATA_ROOT, "Flickr")
    print("  [Dataset] Flickr (89K nodes, Spam Detection)...")
    dataset = Flickr(root=root)
    data = dataset[0]
    # Flickr is 89K — use full data with partition handling
    data = _repackage_graph(data)
    data = _inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.02, m_clique=8)
    return data


def load_reddit():
    """Reddit: 233K nodes, large social network for sybil/troll detection."""
    root = os.path.join(DATA_ROOT, "Reddit")
    print("  [Dataset] Reddit (233K nodes, Sybil/Troll Detection)...")
    dataset = Reddit(root=root)
    data = dataset[0]
    # Reddit is large — subsample to 100K for practical runtime
    data = _maybe_subsample(data, "Reddit", MAX_LARGE_SUBSAMPLE)
    data = _inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.01, m_clique=10)
    return data


# ── Citation / Sanity Check ──────────────────────────────────

def load_planetoid(name):
    """Planetoid datasets (Cora/CiteSeer/PubMed) with injected outliers."""
    root = os.path.join(DATA_ROOT, name)
    print(f"  [Dataset] {name} (Sanity Check)...")
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    data = _inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.03)
    return data


# ==========================================
# 2. Evaluation Engine
# ==========================================

# Models that internally create dense N×N adjacency matrix (via to_dense_adj)
# DLG variants use to_dense_adj in DLGBase.process_graph + dot-product decoder
DENSE_ADJ_MODELS = {"DOMINANT", "AnomalyDAE", "CONAD", "DLG-Base", "DLG"}

# Available system memory limit (bytes) — 24GB with safety margin
MAX_MEMORY_BYTES = 20 * 1024 * 1024 * 1024  # 20GB usable out of 24GB


def _partition_graph(data, partition_size):
    """Split graph into subgraphs using node-chunking.
    Adapted from legacy_adapter._partition_graph — proven on Ethereum graphs.
    """
    from torch_geometric.utils import subgraph as pyg_subgraph
    from torch_geometric.data import Data as PyGData

    num_nodes = data.num_nodes or data.x.size(0)
    if num_nodes <= partition_size:
        return [data]

    indices = torch.arange(num_nodes)
    subgraphs = []
    for i in range(0, num_nodes, partition_size):
        chunk = indices[i:i + partition_size]
        if chunk.numel() == 0:
            continue
        ei, _ = pyg_subgraph(chunk, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        sub = PyGData(x=data.x[chunk].clone(), edge_index=ei, num_nodes=len(chunk))
        if hasattr(data, 'y') and data.y is not None:
            if data.y.numel() == num_nodes:
                sub.y = data.y[chunk].clone()
            else:
                sub.y = data.y.clone()
        sub = _repackage_graph(sub)
        subgraphs.append(sub)
    return subgraphs


def _check_cuda(gpu_id):
    """Robustly check if CUDA GPU is actually usable.
    Uses tensor allocation probe instead of reset_peak_memory_stats,
    which can fail in WSL2/virtualized environments.
    """
    if gpu_id < 0 or not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device=f'cuda:{gpu_id}')
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False

CUDA_AVAILABLE = None

def _estimate_dense_adj_memory(n_nodes):
    """Estimate memory for N×N dense adjacency matrix (float32)."""
    return n_nodes * n_nodes * 4  # 4 bytes per float32

def _skip_result(reason):
    return {k: reason for k in ["ROC-AUC", "PR-AUC", "F1-Score", "Time (s)", "Peak RAM (MB)", "Peak VRAM (MB)"]}

def evaluate_model(model_class, model_name, data, ds_name, is_dlg=False, epoch=50, gpu_id=0):
    global CUDA_AVAILABLE
    print(f"    [{model_name}]", end=" ", flush=True)

    # Lazy one-time GPU check
    if CUDA_AVAILABLE is None:
        CUDA_AVAILABLE = _check_cuda(gpu_id)
        if not CUDA_AVAILABLE and gpu_id >= 0:
            print("\n    ⚠ GPU not usable, falling back to CPU for all models.")

    is_cuda = CUDA_AVAILABLE
    actual_gpu = gpu_id if is_cuda else -1

    if is_cuda:
        try:
            torch.cuda.reset_peak_memory_stats(gpu_id)
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    n_nodes = data.num_nodes

    # ── Dense-adj models: use partition-based approach (from legacy_adapter) ──
    ## For DGraphFin and Yelp, use smaller partition sizes to reduce memory usage
    ## This is because they are more dense than other datasets
    # Per-dataset partition sizes for memory-constrained datasets
    _partition_sizes = {
        "DGraphFin": DGraphFin_PARTITION_SIZE,
        "Yelp":      Yelp_PARTITION_SIZE,
        "Reddit":    Reddit_PARTITION_SIZE,
    }
    current_partition_size = _partition_sizes.get(ds_name, PARTITION_SIZE)

    use_partition = (n_nodes > current_partition_size)

    # ── Adaptive batch size & neighbor sampling ──
    if model_name in DENSE_ADJ_MODELS:
        batch_size = 0   # Full-batch required for dense-adj models
        num_neigh = -1
    else:
        batch_size = 0   # Partitioned graphs are small enough for full-batch
        num_neigh = -1

    process = psutil.Process()
    start_time = time.time()
    import numpy as np

    try:
        if use_partition:
            # ── PARTITION MODE (proven on Ethereum data) ──
            partitions = _partition_graph(data, current_partition_size)
            all_scores = []
            all_labels = []
            nodes_processed = 0
            for part in partitions:
                try:
                    m = model_class(epoch=epoch, gpu=actual_gpu,
                                    batch_size=0, num_neigh=-1, verbose=0)
                except TypeError:
                    m = model_class(epoch=epoch, gpu=actual_gpu,
                                    batch_size=0, verbose=0)
                m.fit(part)
                s = m.decision_function(part)
                s_np = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s)
                all_scores.append(s_np)
                if hasattr(part, 'y') and part.y is not None:
                    all_labels.append(part.y.cpu().numpy())
                
                nodes_processed += part.num_nodes
                percent = min(100.0, (nodes_processed / n_nodes) * 100)
                print(f"\r    [{model_name}] ({len(partitions)} parts) {percent:.1f}% ", end="", flush=True)
                
                del m
                gc.collect()
            print("→ ", end="", flush=True)
            scores_np = np.concatenate(all_scores)
            y_true = np.concatenate(all_labels)
            
            # Handle NaN scores from degenerate partitions
            nan_mask = np.isnan(scores_np) | np.isinf(scores_np)
            if nan_mask.any():
                nan_pct = nan_mask.sum() / len(scores_np) * 100
                print(f"(⚠ {nan_pct:.1f}% NaN scores replaced) ", end="", flush=True)
                scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # ── FULL-GRAPH MODE ──
            try:
                model = model_class(epoch=epoch, gpu=actual_gpu,
                                    batch_size=batch_size, num_neigh=num_neigh, verbose=0)
            except TypeError:
                model = model_class(epoch=epoch, gpu=actual_gpu,
                                    batch_size=batch_size, verbose=0)
            model.fit(data)
            scores = model.decision_function(data)
            scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
            y_true = data.y.cpu().numpy()

        end_time = time.time()
        peak_ram = process.memory_info().rss

        peak_vram = 0
        if is_cuda:
            try:
                peak_vram = torch.cuda.max_memory_allocated(gpu_id)
            except RuntimeError:
                pass

        # Metrics
        roc_auc = roc_auc_score(y_true, scores_np)
        pr_auc = average_precision_score(y_true, scores_np)

        k_ratio = max(sum(y_true) / len(y_true), 0.001)
        threshold_idx = min(int(k_ratio * len(scores_np)), len(scores_np) - 1)
        threshold = sorted(scores_np, reverse=True)[threshold_idx]
        preds = (scores_np >= threshold).astype(int)
        f1 = f1_score(y_true, preds)

        result = {
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1-Score": round(f1, 4),
            "Time (s)": round(end_time - start_time, 2),
            "Peak RAM (MB)": round(peak_ram / (1024 * 1024), 2),
            "Peak VRAM (MB)": round(peak_vram / (1024 * 1024), 2)
        }
        print(f"✓ AUC={result['ROC-AUC']}, T={result['Time (s)']}s")
        return result

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "can't allocate" in err_msg or "cannot allocate" in err_msg:
            print("✗ OOM")
            try: torch.cuda.empty_cache()
            except: pass
            gc.collect()
            return _skip_result("OOM")
        print(f"✗ ERR: {e}")
        return _skip_result("ERR")
    except Exception as e:
        err_msg = str(e).lower()
        if "can't allocate" in err_msg or "cannot allocate" in err_msg:
            print(f"✗ OOM (allocator)")
            gc.collect()
            return _skip_result("OOM")
        print(f"✗ ERR: {e}")
        return _skip_result("ERR")


# ==========================================
# 3. Main Pipeline
# ==========================================
def main():
    gpu_id = 0 if torch.cuda.is_available() else -1

    # ── Dataset Registry ──────────────────────────────────
    # Ordered: Fraud/Financial first, Blockchain/Trust, Social, Citation
    datasets = {
        # ── Fraud / AML / Financial ──
        "Elliptic":    load_elliptic,                            # 46K (full)  Bitcoin AML
        "DGraphFin":   load_dgraphfin,                           # 3.7M (full) Financial Loan Fraud
        "Yelp":        load_yelp,                                # 716K (full) Review Spam
        "Amazon":      load_amazon,                              # 12K (full)  E-commerce Review Fraud
        # ── Blockchain / Trust ──
        "BitcoinOTC":  load_bitcoin_otc,                         # 5.8K (full) Bitcoin Trust Network
        # ── Social Network Anomaly ──
        "Flickr":      load_flickr,                              # 89K (full)  Spam Detection
        "Reddit":      load_reddit,                              # 233K (full) Sybil/Troll
        # ── Citation (Sanity Check) ──
        "Cora":        lambda: load_planetoid("Cora"),           # 2.7K
        "CiteSeer":    lambda: load_planetoid("CiteSeer"),       # 3.3K
        "PubMed":      lambda: load_planetoid("PubMed"),         # 19.7K
    }

    # ── Model Registry (8 models) ─────────────────────────
    models = {
        "DOMINANT":    (DOMINANT,    False),
        "AnomalyDAE":  (AnomalyDAE,  False),
        "CoLA":        (CoLA,        False),
        "CONAD":       (CONAD,       False),
        "GADNR":       (GADNR,       False),
        "OCGNN":       (OCGNN,       False),
        "DLG-Base":    (DLGBase,     True),   # L2 only (no L1 pre-training)
        "DLG":         (DLG,         True),   # Full L1→L2 Decoupled pipeline (ours)
    }

    print("=" * 65)
    print("  SCI Paper Extended Benchmark: DLG vs PyGOD Baselines")
    print(f"  Models: {len(models)} | Datasets: {len(datasets)}")
    print(f"  Data Root: {DATA_ROOT}")
    print("=" * 65)

    # ── RESUME / LOGGING LOGIC ────────────────────────────
    completed_datasets = set()
    completed_pairs = set()       # (dataset, model) pairs already evaluated
    failed_datasets = set()
    csv_path = os.path.join(REPORT_DIR, "benchmark_8x10_results.csv")
    fail_log_path = os.path.join(REPORT_DIR, "failed_datasets.log")

    if os.path.exists(fail_log_path):
        with open(fail_log_path, 'r') as f:
            failed_datasets = set(line.strip() for line in f if line.strip())

    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            results_list = df_existing.to_dict('records')
            
            # Build set of completed (dataset, model) pairs for fine-grained skip
            for _, row in df_existing.iterrows():
                completed_pairs.add((row['Dataset'], row['Model']))
            
            # Check which datasets have all models evaluated
            for ds in df_existing['Dataset'].unique():
                ds_models = df_existing[df_existing['Dataset'] == ds]['Model'].nunique()
                if ds_models >= len(models):
                    completed_datasets.add(ds)
            
            n_partial = len(completed_pairs) - len(completed_datasets) * len(models)
            print(f"  [Resume] Loaded {len(results_list)} previous results.")
            print(f"  [Resume] Completed datasets (will skip entirely): {', '.join(completed_datasets) if completed_datasets else 'None'}")
            if n_partial > 0:
                partial_ds = {ds for ds, _ in completed_pairs if ds not in completed_datasets}
                for pds in partial_ds:
                    done_models = [m for d, m in completed_pairs if d == pds]
                    print(f"  [Resume] {pds}: {len(done_models)}/{len(models)} models done → will skip: {', '.join(done_models)}")
            print(f"  [Resume] Failed datasets (will skip): {', '.join(failed_datasets) if failed_datasets else 'None'}")
        except Exception as e:
            print(f"  [Resume] Failed to parse existing CSV: {e}")
            results_list = []
    else:
        results_list = []

    for ds_name, ds_loader in datasets.items():
        print(f"\n{'─' * 65}")
        
        if ds_name in completed_datasets:
            print(f"  📊 Dataset: {ds_name} [ALREADY BENCHMARKED - SKIPPING]")
            print(f"{'─' * 65}")
            continue
            
        if ds_name in failed_datasets:
            print(f"  📊 Dataset: {ds_name} [PREVIOUSLY FAILED - SKIPPING]")
            print(f"{'─' * 65}")
            continue
            
        print(f"  📊 Dataset: {ds_name}")
        print(f"{'─' * 65}")

        try:
            data = ds_loader()
        except Exception as e:
            print(f"  [SKIP] Failed to load {ds_name}: {e}")
            traceback.print_exc()
            with open(fail_log_path, 'a') as f:
                f.write(f"{ds_name}\n")
            continue

        if data is None:
            # Also log as failed so we don't retry downloading next time
            with open(fail_log_path, 'a') as f:
                f.write(f"{ds_name}\n")
            continue

        # Count anomalies correctly (handle Elliptic where y∈{0,1,2})
        if hasattr(data, 'eval_mask') and data.eval_mask is not None:
            y_masked = data.y[data.eval_mask]
            n_anomalies = int((y_masked == 1).sum().item())
            n_total = int(data.eval_mask.sum().item())
        else:
            n_anomalies = int((data.y == 1).sum().item())
            n_total = data.num_nodes
        print(f"  Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | "
              f"Anomalies: {n_anomalies:,} ({n_anomalies/n_total*100:.1f}%)")

        for model_name, (model_class, is_dlg) in models.items():
            # Skip models already evaluated for this dataset (fine-grained resume)
            if (ds_name, model_name) in completed_pairs:
                print(f"    [{model_name}] ⏭ already evaluated — skipping")
                continue

            res = evaluate_model(model_class, model_name, data, ds_name=ds_name, is_dlg=is_dlg, epoch=50, gpu_id=gpu_id)
            res["Dataset"] = ds_name
            res["Model"] = model_name
            res["Nodes"] = data.num_nodes
            results_list.append(res)

            gc.collect()
            try:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except: pass

        # Save intermediate results after each dataset (in case of crash)
        _save_results(results_list)

        # ── CUDA recovery: detect & recover from corrupted GPU state ──
        # A single CUDA illegal memory access corrupts ALL subsequent GPU ops
        # in the same process. We detect this and reset.
        if torch.cuda.is_available():
            try:
                _probe = torch.zeros(1, device='cuda:0')
                del _probe
            except RuntimeError:
                print("  ⚠ CUDA state corrupted — resetting GPU device...")
                try:
                    torch.cuda.synchronize()
                except:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                # Force re-check on next evaluate_model call
                global CUDA_AVAILABLE
                CUDA_AVAILABLE = None
                # In some cases, the only way to recover is to skip to CPU
                # for the remaining datasets. We try GPU first.
                try:
                    _probe2 = torch.zeros(1, device='cuda:0')
                    del _probe2
                    print("  ✓ CUDA recovered successfully.")
                except RuntimeError:
                    print("  ✗ CUDA unrecoverable. Remaining datasets will use CPU.")
                    CUDA_AVAILABLE = False
                    gpu_id = -1

    # Final save
    _save_results(results_list, final=True)


def _save_results(results_list, final=False):
    """Save results to CSV, overwriting each time."""
    df = pd.DataFrame(results_list)
    cols = ["Dataset", "Nodes", "Model", "ROC-AUC", "PR-AUC", "F1-Score",
            "Time (s)", "Peak RAM (MB)", "Peak VRAM (MB)"]
    df = df[[c for c in cols if c in df.columns]]

    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(REPORT_DIR, "benchmark_8x10_results.csv")
    df.to_csv(csv_path, index=False)

    if final:
        print(f"\n{'=' * 65}")
        print("                   BENCHMARK COMPLETE")
        print(f"{'=' * 65}")
        print(df.to_string(index=False))
        print(f"\n📁 Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
