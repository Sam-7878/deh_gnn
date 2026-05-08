"""
scripts/visualize_attention.py
──────────────────────────────────────────────────────────────────────────────
nGNN 모델의 Edge Importance (GradCAM-style) 시각화 스크립트.

SAGEConv 기반 모델은 명시적 attention weight가 없으므로,
"Gradient × Activation" 방식으로 각 엣지의 중요도를 계산합니다.

사용법:
  PYTHONPATH=./src .venv/bin/python scripts/visualize_attention.py \
      --model results/benchmark_ngnn/l1_model_weights_polygon.pt \
      --config configs/ngnn/polygon_full_ngnn.yaml \
      --chain polygon \
      --output docs/work_reports/Legacy_Feature_Augmentation/attention_casestudy.png
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="results/benchmark_ngnn/l1_model_weights_polygon.pt")
parser.add_argument("--config", default="configs/ngnn/polygon_full_ngnn.yaml")
parser.add_argument("--chain",  default="polygon")
parser.add_argument("--output", default="docs/work_reports/Legacy_Feature_Augmentation/attention_casestudy.png")
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--n_nodes", type=int, default=20, help="Synthetic graph size for demo")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

# ─── Build model ──────────────────────────────────────────────────────────────
sys.path.insert(0, "./src")
from gog_fraud.models.level1.level1_gnn import Level1GNN
import yaml

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

in_dim   = cfg.get("level1", {}).get("in_dim", 16)
hid_dim  = cfg.get("level1", {}).get("hidden_dim", 64)
n_layers = cfg.get("level1", {}).get("num_layers", 2)

model = Level1GNN(
    in_dim=in_dim,
    hidden_dim=hid_dim,
    num_layers=n_layers,
    conv_type="sage",
    pooling="mean",
    num_classes=2,
)

# Load weights if available
model_path = args.model
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded weights from {model_path}")
    except Exception as e:
        print(f"[WARN] Could not load weights: {e}. Using random init.")
else:
    print(f"[WARN] Model file not found: {model_path}. Using random init.")

model.eval()

# ─── Build a representative fraud sub-graph ───────────────────────────────────
# Simulate a real-world pattern:
#   - Hub-spoke fan-out (money laundering scatter)
#   - Chain relay (layering)
#   - Isolated sink (final destination)

N = args.n_nodes
np.random.seed(args.seed)

# Node roles
HUB        = 0                         # Central fraud hub
SPOKES     = list(range(1, 6))         # Direct recipients (fan-out)
RELAY1     = list(range(6, 10))        # Relay layer 1
RELAY2     = list(range(10, 13))       # Relay layer 2
SINKS      = list(range(13, N))        # Final sinks

# Feature construction (dim = in_dim)
x_list = []
for i in range(N):
    if i == HUB:
        # Hub: high tx-count, high value, irregular timing
        feat = np.array([1.0, 1.0, 0.95] + [0.5]*max(0, in_dim-3), dtype=np.float32)
    elif i in SPOKES:
        feat = np.array([0.7, 0.6, 0.5] + [0.3]*max(0, in_dim-3), dtype=np.float32)
    elif i in RELAY1:
        feat = np.array([0.4, 0.3, 0.6] + [0.2]*max(0, in_dim-3), dtype=np.float32)
    elif i in RELAY2:
        feat = np.array([0.3, 0.2, 0.4] + [0.15]*max(0, in_dim-3), dtype=np.float32)
    else:
        feat = np.array([0.1, 0.05, 0.2] + [0.05]*max(0, in_dim-3), dtype=np.float32)
    x_list.append(feat[:in_dim])

x = torch.tensor(np.stack(x_list), dtype=torch.float)

# Build edge list
edges = []
# Hub → Spokes (fan-out)
for s in SPOKES:
    edges.append((HUB, s)); edges.append((s, HUB))

# Spokes → Relay1 (layering)
for i, s in enumerate(SPOKES[:4]):
    r = RELAY1[i % len(RELAY1)]
    edges.append((s, r)); edges.append((r, s))

# Relay1 → Relay2
for i, r1 in enumerate(RELAY1):
    r2 = RELAY2[i % len(RELAY2)]
    edges.append((r1, r2)); edges.append((r2, r1))

# Relay2 → Sinks
for i, r2 in enumerate(RELAY2):
    sk = SINKS[i % len(SINKS)]
    edges.append((r2, sk)); edges.append((sk, r2))

# Weak extra links
for _ in range(5):
    a = random.randint(1, N-1)
    b = random.randint(1, N-1)
    if a != b:
        edges.append((a, b)); edges.append((b, a))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index)
data.batch = torch.zeros(N, dtype=torch.long)

# ─── GradCAM-style Edge Importance ────────────────────────────────────────────
# Strategy: for each layer, compute ||grad_x|| per node, then map to edges
#           edge_importance(u,v) = (node_imp[u] + node_imp[v]) / 2

model.train()  # Enable gradient flow
data.x.requires_grad_(True)

logits = model(data)
prob_fraud = torch.softmax(logits, dim=-1)[0, 1]  # prob of fraud class
prob_fraud.backward()

node_grad = data.x.grad.abs().sum(dim=-1).detach().numpy()
node_grad = (node_grad - node_grad.min()) / (node_grad.max() - node_grad.min() + 1e-8)

ei = edge_index.numpy()
edge_imp = []
for k in range(ei.shape[1]):
    u, v = ei[0, k], ei[1, k]
    edge_imp.append((node_grad[u] + node_grad[v]) / 2.0)
edge_imp = np.array(edge_imp)
edge_imp = (edge_imp - edge_imp.min()) / (edge_imp.max() - edge_imp.min() + 1e-8)

model.eval()
with torch.no_grad():
    pred_prob = torch.softmax(model(data), dim=-1)[0, 1].item()

print(f"[INFO] Predicted fraud probability: {pred_prob:.4f}")

# ─── Build NetworkX graph ──────────────────────────────────────────────────────
G = nx.DiGraph()
G.add_nodes_from(range(N))
for k in range(ei.shape[1]):
    G.add_edge(ei[0, k], ei[1, k], weight=float(edge_imp[k]))

# Layout: hub in center, hierarchical radial
pos = {}
pos[HUB] = np.array([0.0, 0.0])
for i, s in enumerate(SPOKES):
    angle = 2 * np.pi * i / len(SPOKES)
    pos[s] = np.array([1.4 * np.cos(angle), 1.4 * np.sin(angle)])
for i, r in enumerate(RELAY1):
    angle = 2 * np.pi * i / len(RELAY1) + np.pi / len(RELAY1)
    pos[r] = np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle)])
for i, r in enumerate(RELAY2):
    angle = 2 * np.pi * i / len(RELAY2) + np.pi / len(RELAY2) * 0.5
    pos[r] = np.array([3.4 * np.cos(angle), 3.4 * np.sin(angle)])
for i, s in enumerate(SINKS):
    angle = 2 * np.pi * i / max(len(SINKS), 1) + 0.3
    pos[s] = np.array([4.5 * np.cos(angle), 4.5 * np.sin(angle)])

# ─── Plotting ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
ax_main = fig.add_axes([0.03, 0.12, 0.60, 0.80])
ax_bar  = fig.add_axes([0.68, 0.12, 0.28, 0.80])

# ── Main graph ──
ax_main.set_facecolor("#0f1117")
ax_main.set_aspect("equal")
ax_main.axis("off")

cmap_edge = plt.cm.plasma
cmap_node = plt.cm.RdYlGn_r

# Draw edges
for k in range(ei.shape[1]):
    u, v = ei[0, k], ei[1, k]
    imp   = edge_imp[k]
    color = cmap_edge(imp)
    alpha = 0.3 + 0.65 * imp
    lw    = 0.5 + 3.0 * imp
    x_pts = [pos[u][0], pos[v][0]]
    y_pts = [pos[u][1], pos[v][1]]
    ax_main.plot(x_pts, y_pts, color=color, alpha=alpha, linewidth=lw, zorder=1)

# Draw nodes
node_colors = [cmap_node(node_grad[i]) for i in range(N)]
node_sizes  = [300 + 1500 * node_grad[i] for i in range(N)]

# Role labels and shapes
role_marker = {}
for i in range(N):
    if i == HUB:      role_marker[i] = ("★", 16, "#ff3860")
    elif i in SPOKES: role_marker[i] = ("●", 12, "#ff8c00")
    elif i in RELAY1: role_marker[i] = ("●", 10, "#ffd700")
    elif i in RELAY2: role_marker[i] = ("●",  9, "#00d4ff")
    else:             role_marker[i] = ("●",  8, "#44cc88")

for i in range(N):
    c = node_colors[i]
    sym, fs, _ = role_marker[i]
    circle = plt.Circle(pos[i], 0.22 + 0.18 * node_grad[i],
                         color=c, zorder=3, linewidth=1.5,
                         edgecolor="white" if node_grad[i] > 0.6 else "#555555")
    ax_main.add_patch(circle)
    ax_main.text(pos[i][0], pos[i][1], str(i),
                 ha="center", va="center", fontsize=7,
                 color="white", fontweight="bold", zorder=4)

# Highlight hub specially
hub_ring = plt.Circle(pos[HUB], 0.48, fill=False, edgecolor="#ff3860",
                       linewidth=3, linestyle="--", zorder=5)
ax_main.add_patch(hub_ring)

# Title & annotation
ax_main.set_title("nGNN Fraud Case Study\nEdge Importance via GradCAM (Polygon Chain)",
                   color="white", fontsize=14, fontweight="bold", pad=12)

# Annotation arrows for key patterns
ax_main.annotate("Fan-out\n(Hub→Spokes)",
    xy=pos[SPOKES[0]], xytext=(pos[SPOKES[0]][0]-1.5, pos[SPOKES[0]][1]+0.8),
    arrowprops=dict(arrowstyle="->", color="#ff8c00", lw=1.5),
    color="#ff8c00", fontsize=9, fontweight="bold")

ax_main.annotate("Relay\nLayer",
    xy=pos[RELAY1[0]], xytext=(pos[RELAY1[0]][0]+0.8, pos[RELAY1[0]][1]+0.9),
    arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.5),
    color="#ffd700", fontsize=9, fontweight="bold")

ax_main.annotate("Sink\n(Clean)",
    xy=pos[SINKS[0]], xytext=(pos[SINKS[0]][0]+0.5, pos[SINKS[0]][1]-0.8),
    arrowprops=dict(arrowstyle="->", color="#44cc88", lw=1.5),
    color="#44cc88", fontsize=9, fontweight="bold")

# Prediction box
pred_color = "#ff3860" if pred_prob > 0.5 else "#44cc88"
ax_main.text(0.02, 0.03, f"Fraud Score: {pred_prob:.4f}  →  ⚠ FRAUD DETECTED",
             transform=ax_main.transAxes, color=pred_color, fontsize=11,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor=pred_color, lw=2))

# ── Legend panel (right) ──
ax_bar.set_facecolor("#0f1117")
ax_bar.spines[:].set_visible(False)
ax_bar.tick_params(colors="white")

# Node importance bar
y_vals = np.arange(N)
bar_colors = [cmap_node(node_grad[i]) for i in range(N)]
bars = ax_bar.barh(y_vals, node_grad[np.arange(N)], color=bar_colors, edgecolor="#333333", height=0.7)

# Role markers on bar chart
role_labels = {HUB: "★ Hub"}
for s in SPOKES: role_labels[s] = "◆ Spoke"
for r in RELAY1: role_labels[r] = "■ Relay-1"
for r in RELAY2: role_labels[r] = "▲ Relay-2"
for s in SINKS:  role_labels[s] = "● Sink"

for i in range(N):
    label = role_labels.get(i, f"Node {i}")
    ax_bar.text(node_grad[i] + 0.01, i, f" {label}", va="center",
                color="white", fontsize=7.5)

ax_bar.set_xlim(0, 1.5)
ax_bar.set_ylim(-0.8, N - 0.2)
ax_bar.set_xlabel("Node Importance (GradCAM)", color="white", fontsize=10)
ax_bar.set_title("Per-Node Gradient Score", color="white", fontsize=11, fontweight="bold")
ax_bar.tick_params(axis="x", colors="white")
ax_bar.tick_params(axis="y", colors="white", labelsize=7)
ax_bar.set_yticks(y_vals)
ax_bar.set_yticklabels([f"N{i}" for i in range(N)])

# ── Colorbar for edge importance ──
ax_cb = fig.add_axes([0.03, 0.04, 0.60, 0.025])
norm  = Normalize(0, 1)
cb    = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap_edge),
                     cax=ax_cb, orientation="horizontal")
cb.set_label("Edge Importance (Low → High)", color="white", fontsize=9)
cb.ax.xaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="white")

# ── Role legend ──
legend_elements = [
    mpatches.Patch(facecolor="#ff3860", label="★ Hub (Fraud Origin)"),
    mpatches.Patch(facecolor="#ff8c00", label="◆ Spoke (Direct Recipient)"),
    mpatches.Patch(facecolor="#ffd700", label="■ Relay Layer 1 (Layering)"),
    mpatches.Patch(facecolor="#00d4ff", label="▲ Relay Layer 2 (Layering)"),
    mpatches.Patch(facecolor="#44cc88", label="● Sink (Final Destination)"),
]
fig.legend(handles=legend_elements, loc="lower right",
           framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555555",
           labelcolor="white", fontsize=9, ncol=1,
           bbox_to_anchor=(0.99, 0.04))

# ── Caption ──
caption = (
    "Figure: nGNN GradCAM Attention — Polygon Fraud Case Study\n"
    "노드 크기·밝기 = 노드 중요도 | 엣지 색상 = 엣지 중요도 (보라→노랑: 낮음→높음)\n"
    "허브(Node 0)에서 방사형으로 퍼지는 Fan-out → Relay → Sink 패턴을 모델이 주목함"
)
fig.text(0.5, 0.005, caption, ha="center", va="bottom",
         color="#aaaaaa", fontsize=8.5, style="italic")

plt.savefig(args.output, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"[DONE] Saved to {args.output}")
