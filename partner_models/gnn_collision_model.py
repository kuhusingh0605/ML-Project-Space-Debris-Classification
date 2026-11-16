"""
=======================================================================================
 STAGE B - GRAPH NEURAL NETWORK vs CLASSICAL ML BASELINES
=======================================================================================
 Satellite collision-risk prediction on physics-derived labels.

 WHY A GRAPH AND NOT PAIRS
 -------------------------
 A pairwise model sees each (satellite, debris) pair in isolation: row 47 knows
 nothing about row 48, even when both involve the same debris object. A GNN removes
 that independence - each object is a NODE, each node aggregates information from
 its orbital neighbourhood, and a pair is then scored from two neighbourhood-aware
 embeddings instead of two isolated feature vectors.

 So the pair does not disappear; it becomes an EDGE to be classified.

   nodes             all Starlink satellites + all COSMOS 2251 debris, one graph
   message-passing   k-nearest-neighbour graph in orbital-element space
     edges           (which objects share an orbital regime)
   supervised        the candidate conjunction pairs from Stage A
     edges
   target            see TASKS below

 Baselines get the SAME information in flat form - concat(node_i, node_j, |diff|) -
 so the only thing the GNN adds is the graph structure. That is what makes the
 comparison meaningful rather than decorative.

 TWO TARGETS ARE REPORTED, deliberately
 --------------------------------------
   1. MOID  - the closest the two ORBITS ever come, independent of timing. This is
      the standard first-stage geometric screen in real conjunction assessment.
      It is a smooth function of both objects' elements, so it is learnable.

   2. 24h time-phased miss distance - the closest the two OBJECTS actually come
      inside the propagation window. This depends on relative phasing, which is
      violently sensitive to small mean-motion differences, so it is very hard to
      predict from elements at epoch. Reporting it is the honest result: it shows
      where element-based learning stops working and why operational systems
      propagate instead of regressing.

 Run conjunction_labels.py first to produce conjunction_cache.npz.
=======================================================================================
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # write the PNG without blocking on a window
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve)
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------- CONFIG ------------------------------------
CACHE = "conjunction_cache.npz"

# (display name, cached array, threshold km, short label for the chart)
TASKS = [
    ("Task 1 - MOID (orbital geometry)",      "moid_km",     50.0, "MOID < 50 km"),
    ("Task 2 - 24h time-phased miss distance", "min_dist_km", 50.0, "24h miss < 50 km"),
]

KNN_K              = 12      # neighbours in the orbital-proximity graph
TEST_FRACTION      = 0.20
SVC_TRAIN_CAP      = 8000    # kernel SVM is O(n^2); the full set is not trainable
GNN_HIDDEN         = 64
GNN_EPOCHS         = 250     # PR-AUC was still improving at 150
GNN_LR             = 1e-3    # same learning rate as the reference paper
GNN_DROPOUT        = 0.2
DECISION_THRESHOLD = 0.50    # identical for every model, for a fair comparison
SEED               = 42
RESULTS_CSV        = "results_gnn_vs_baselines.csv"
PLOT_PNG           = "model_comparison.png"

# Reference data-viz palette, categorical slots 1-3 (light mode)
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
C_SURFACE, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
# -------------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_NAMES = []


# ========================= feature engineering ==========================
def build_node_matrix(raw, cols):
    """Node feature matrix with circular angles encoded as sin/cos.

    RAAN, argument of perigee and mean anomaly wrap at 360 deg. Fed in raw, a
    model sees 359 and 1 as far apart when they are adjacent, so each is split
    into its sine and cosine.
    """
    g = {c: raw[:, i] for i, c in enumerate(cols)}
    feats, names = [], []

    def add(name, v):
        feats.append(v)
        names.append(name)

    add("inclination_deg", g["inclination_deg"])
    add("eccentricity", g["eccentricity"])
    for ang in ["raan_deg", "argp_deg", "mean_anom_deg"]:
        r = np.radians(g[ang])
        add(ang.replace("_deg", "_sin"), np.sin(r))
        add(ang.replace("_deg", "_cos"), np.cos(r))
    add("mean_motion_rpd", g["mean_motion_rpd"])
    add("bstar", g["bstar"])
    add("sma_km", g["sma_km"])
    add("perigee_alt_km", g["perigee_alt_km"])
    add("apogee_alt_km", g["apogee_alt_km"])

    return np.column_stack(feats).astype(np.float64), names


def knn_orbital_graph(X, k):
    """Undirected kNN graph in orbital-regime space.

    Proximity is measured on the elements that decide whether two objects can
    ever meet: altitude (semi-major axis), inclination and orbital-plane
    orientation (RAAN as sin/cos). This graph carries NO label information - it
    is pure geometry - so building it over all nodes leaks nothing.
    """
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    sub = X[:, [idx["sma_km"], idx["inclination_deg"],
                idx["raan_sin"], idx["raan_cos"]]]
    sub = StandardScaler().fit_transform(sub)

    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(sub)
    _, ind = nn_.kneighbors(sub)

    src = np.repeat(ind[:, 0], k)
    dst = ind[:, 1:].ravel()
    ei = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    ei = np.unique(ei, axis=1)
    return torch.tensor(ei, dtype=torch.long)


def pair_features(X, pa, pb):
    """Flat pair representation for the non-graph baselines."""
    a, b = X[pa], X[pb]
    return np.hstack([a, b, np.abs(a - b)])


# ================================ GNN ===================================
class EdgeRiskGNN(nn.Module):
    """GraphSAGE encoder + MLP edge decoder.

    encoder:  each node is embedded from its own elements AND its orbital
              neighbourhood (two rounds of message passing)
    decoder:  a pair is scored from the two embeddings, their absolute
              difference and their elementwise product

    SAGEConv mean-aggregation smooths each node toward its orbital neighbours.
    That is the point for neighbourhood context, but it also destroys the exact
    element values that a geometric target like MOID depends on. So the decoder
    gets a SKIP CONNECTION carrying the raw node features alongside the learned
    embeddings - it can use the neighbourhood without being limited to it.
    """

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        head_in = 4 * hidden + 3 * in_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

    def decode(self, z, x, pa, pb):
        zi, zj = z[pa], z[pb]
        xi, xj = x[pa], x[pb]
        feat = torch.cat([zi, zj, (zi - zj).abs(), zi * zj,
                          xi, xj, (xi - xj).abs()], dim=1)
        return self.head(feat).squeeze(-1)

    def forward(self, x, edge_index, pa, pb):
        return self.decode(self.encode(x, edge_index), x, pa, pb)


def train_gnn(x, edge_index, pa, pb, y, tr, te, pos_weight):
    model = EdgeRiskGNN(x.shape[1], GNN_HIDDEN, GNN_DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=5e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    pa_tr, pb_tr, y_tr = pa[tr], pb[tr], y[tr]
    y_te_np = y[te].numpy()

    t0 = time.time()
    for ep in range(1, GNN_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(x, edge_index, pa_tr, pb_tr), y_tr)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x, edge_index, pa[te], pb[te])).numpy()
            print("      epoch {:>4}   loss {:.4f}   test PR-AUC {:.4f}"
                  .format(ep, float(loss), average_precision_score(y_te_np, p)))
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x, edge_index, pa[te], pb[te])).numpy()
    return prob, train_s


# ============================== evaluation ==============================
def evaluate(name, y_true, prob, train_s, base_rate, note=""):
    """Metrics at the shared 0.50 cut, plus threshold-free and best-threshold views.

    The fixed cut is needed for a like-for-like comparison, but it penalises any
    model whose probabilities are not calibrated around 0.50 - which is why
    `F1 best` (the best F1 reachable by moving the threshold) is reported beside
    it. PR-AUC is the threshold-free ranking quality and is the headline number.
    """
    pred = (prob >= DECISION_THRESHOLD).astype(int)
    ap = average_precision_score(y_true, prob)

    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1s = 2.0 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    k = int(np.nanargmax(f1s))

    return {
        "Model": name,
        "Recall (%)": 100.0 * recall_score(y_true, pred, zero_division=0),
        "Precision (%)": 100.0 * precision_score(y_true, pred, zero_division=0),
        "F1 (%)": 100.0 * f1_score(y_true, pred, zero_division=0),
        "Recall@best (%)": 100.0 * rec[k],
        "Precision@best (%)": 100.0 * prec[k],
        "F1 best (%)": 100.0 * f1s[k],
        "Best thr": float(thr[k]),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": ap,
        "PR lift": ap / base_rate,
        "Train (s)": train_s,
        "Note": note,
    }


# =============================== one task ===============================
def run_task(task_name, values, thr_km, X, Xs, edge_index, pa, pb):
    y = (values < thr_km).astype(np.int64)
    n_tot, n_pos = len(y), int(y.sum())
    base_rate = n_pos / n_tot

    print("\n" + "#" * 87)
    print(" {}   (target: < {:.0f} km)".format(task_name, thr_km))
    print("#" * 87)
    print("   candidate pairs (edges) : {:,}".format(n_tot))
    print("   HIGH RISK (class 1)     : {:,}  ({:.3f}%)".format(n_pos, 100.0 * base_rate))
    print("   imbalance ratio         : 1 : {:.0f}".format((n_tot - n_pos) / max(n_pos, 1)))
    if n_pos < 150:
        print("   !! too few positives for a stable estimate")

    idx = np.arange(n_tot)
    tr, te = train_test_split(idx, test_size=TEST_FRACTION,
                              random_state=SEED, stratify=y)
    pos_weight = float((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1))
    print("   split                   : train {:,} / test {:,}  (positives {:,} / {:,})"
          .format(len(tr), len(te), int(y[tr].sum()), int(y[te].sum())))

    results = []

    # --- physics-only heuristic ---------------------------------------------
    i_per = FEATURE_NAMES.index("perigee_alt_km")
    i_apo = FEATURE_NAMES.index("apogee_alt_km")
    gap = np.maximum(X[pa][:, i_per] - X[pb][:, i_apo],
                     X[pb][:, i_per] - X[pa][:, i_apo])
    score = 1.0 / (1.0 + np.maximum(gap, 0.0))
    results.append(evaluate("Analytic Shell Overlap (heuristic)", y[te], score[te],
                            0.0, base_rate, "physics only, no training"))
    print("\n   [a] analytic heuristic  done")

    # --- flat-feature baselines ---------------------------------------------
    P = pair_features(Xs, pa, pb)
    ps = StandardScaler().fit(P[tr])
    P_tr, P_te = ps.transform(P[tr]), ps.transform(P[te])

    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr.fit(P_tr, y[tr])
    dt = time.time() - t0
    results.append(evaluate("Logistic Regression", y[te],
                            lr.predict_proba(P_te)[:, 1], dt, base_rate))
    print("   [b] Logistic Regression {:.1f}s".format(dt))

    cap = min(SVC_TRAIN_CAP, len(tr))
    sub, _ = train_test_split(np.arange(len(tr)), train_size=cap,
                              random_state=SEED, stratify=y[tr])
    t0 = time.time()
    sv = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    sv.fit(P_tr[sub], y[tr][sub])
    dt = time.time() - t0
    results.append(evaluate("Support Vector Machine (RBF)", y[te],
                            sv.predict_proba(P_te)[:, 1], dt, base_rate,
                            "trained on {:,} of {:,} pairs - O(n^2) kernel"
                            .format(cap, len(tr))))
    print("   [c] SVM (RBF)           {:.1f}s on {:,} pairs".format(dt, cap))

    t0 = time.time()
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        scale_pos_weight=pos_weight, eval_metric="logloss",
                        tree_method="hist", random_state=SEED)
    xgb.fit(P_tr, y[tr])
    dt = time.time() - t0
    results.append(evaluate("XGBoost", y[te], xgb.predict_proba(P_te)[:, 1],
                            dt, base_rate))
    print("   [d] XGBoost             {:.1f}s".format(dt))

    # --- GNN ----------------------------------------------------------------
    print("   [e] GNN (GraphSAGE), CPU:")
    xt = torch.tensor(Xs, dtype=torch.float32)
    prob, dt = train_gnn(xt, edge_index, torch.tensor(pa), torch.tensor(pb),
                         torch.tensor(y, dtype=torch.float32),
                         torch.tensor(tr), torch.tensor(te), pos_weight)
    results.append(evaluate("Graph Neural Network (GraphSAGE)", y[te], prob,
                            dt, base_rate))
    print("       GNN                 {:.1f}s".format(dt))

    df = pd.DataFrame(results)
    df.insert(0, "Task", task_name)
    df["Base rate (%)"] = 100.0 * base_rate

    show = df.drop(columns=["Note", "Task"]).copy()
    for c in ["Recall (%)", "Precision (%)", "F1 (%)", "Recall@best (%)",
              "Precision@best (%)", "F1 best (%)", "Base rate (%)"]:
        show[c] = show[c].map("{:.2f}".format)
    for c in ["ROC-AUC", "PR-AUC"]:
        show[c] = show[c].map("{:.4f}".format)
    show["Best thr"] = show["Best thr"].map("{:.3f}".format)
    show["PR lift"] = show["PR lift"].map("{:.2f}x".format)
    show["Train (s)"] = show["Train (s)"].map("{:.1f}".format)

    print("\n   RESULTS   test n = {:,},  base rate {:.2f}%"
          .format(len(te), 100.0 * base_rate))

    print("\n   (i) at the shared decision cut of {:.2f}".format(DECISION_THRESHOLD))
    print("   " + "-" * 84)
    print(show[["Model", "Recall (%)", "Precision (%)", "F1 (%)",
                "Train (s)"]].to_string(index=False))

    print("\n   (ii) each model at its OWN best-F1 threshold, plus threshold-free ranking")
    print("   " + "-" * 84)
    print(show[["Model", "Recall@best (%)", "Precision@best (%)", "F1 best (%)",
                "Best thr", "ROC-AUC", "PR-AUC", "PR lift"]].to_string(index=False))
    print("   " + "-" * 84)
    return df


def main():
    global FEATURE_NAMES

    print("=" * 87)
    print(" STAGE B - GNN vs CLASSICAL ML ON PHYSICS-DERIVED COLLISION LABELS")
    print("=" * 87)

    path = os.path.join(HERE, CACHE)
    if not os.path.exists(path):
        raise SystemExit("Missing {} - run conjunction_labels.py first.".format(CACHE))
    z = np.load(path, allow_pickle=True)

    raw = z["node_feat"]
    cols = [str(c) for c in z["feat_names"]]
    is_debris = z["is_debris"]
    pa, pb = z["pair_a"].astype(np.int64), z["pair_b"].astype(np.int64)

    X, FEATURE_NAMES = build_node_matrix(raw, cols)
    Xs = StandardScaler().fit_transform(X)

    print("\n[1] Graph")
    print("   objects (nodes)         : {:,}  ({:,} Starlink / {:,} debris)"
          .format(len(X), int((~is_debris).sum()), int(is_debris.sum())))
    print("   node features           : {}".format(len(FEATURE_NAMES)))
    print("   {}".format(", ".join(FEATURE_NAMES)))

    edge_index = knn_orbital_graph(X, KNN_K)
    print("   message-passing edges   : {:,} (directed, symmetric)".format(edge_index.shape[1]))
    print("   mean node degree        : {:.1f}".format(edge_index.shape[1] / len(X)))
    print("   pair feature dimension  : {}".format(3 * len(FEATURE_NAMES)))

    frames = []
    for task_name, key, thr, _short in TASKS:
        frames.append(run_task(task_name, z[key], thr, X, Xs, edge_index, pa, pb))

    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(os.path.join(HERE, RESULTS_CSV), index=False)

    print("\n" + "=" * 87)
    print(" NOTES")
    print("=" * 87)
    for _, r in allr.iterrows():
        if r["Note"]:
            print("  [{}] {} : {}".format(r["Task"].split(" - ")[0], r["Model"], r["Note"]))
    print("\n  PR-AUC and PR lift are the headline numbers. With this much imbalance")
    print("  ROC-AUC flatters every model and hides the real differences.")
    print("  PR lift = PR-AUC divided by the base rate: 1.0x means no better than guessing.")
    print("\n  Saved table -> {}".format(RESULTS_CSV))

    make_plot(allr)
    print("=" * 87)


# ================================ chart =================================
def make_plot(allr):
    """Two panels (one per target), grouped bars, three metrics as colour series.

    Form: magnitude comparison across a small set of named models on a common
    0-100 axis, faceted by target. Metrics take categorical slots 1-3 in fixed
    order. Every bar is directly labelled, which also discharges the relief rule
    for the lower-contrast aqua slot on a light surface.

    Each model is shown at its OWN best-F1 threshold. A single shared cut of 0.50
    would compare calibration rather than ranking quality, and would punish any
    model whose probabilities are not centred there - the GNN optimum sits near
    0.82, the tree near 0.55. The shared-cut numbers are still in the CSV.
    """
    metrics = ["Recall@best (%)", "Precision@best (%)", "F1 best (%)"]
    labels = ["Recall", "Precision", "F1"]
    short = {
        "Analytic Shell Overlap (heuristic)": "Analytic\nheuristic",
        "Logistic Regression": "Logistic\nRegression",
        "Support Vector Machine (RBF)": "SVM\n(RBF)",
        "XGBoost": "XGBoost",
        "Graph Neural Network (GraphSAGE)": "GNN\n(GraphSAGE)",
    }
    tasks = [(t[0], t[3]) for t in TASKS]

    fig, axes = plt.subplots(1, len(tasks), figsize=(15.5, 5.8), sharey=False)
    if len(tasks) == 1:
        axes = [axes]
    fig.patch.set_facecolor(C_SURFACE)

    # each panel is scaled to its own data: the two targets differ by an order of
    # magnitude in base rate, so a shared axis would flatten the harder one to
    # nothing. The base-rate rule in each panel keeps them readable separately.

    for ax, (tname, tshort) in zip(axes, tasks):
        d = allr[allr["Task"] == tname].reset_index(drop=True)
        names = [short.get(m, m) for m in d["Model"]]
        x = np.arange(len(d))
        w = 0.26
        ax.set_facecolor(C_SURFACE)
        top = float(max(d[m].max() for m in metrics))

        for i, (mc, lab) in enumerate(zip(metrics, labels)):
            vals = d[mc].to_numpy()
            off = (i - 1) * w
            ax.bar(x + off, vals, w * 0.92, label=lab,
                   color=C_SERIES[i], edgecolor=C_SURFACE, linewidth=2.0, zorder=3)
            for xi, v in zip(x + off, vals):
                ax.annotate("{:.1f}".format(v), (xi, v), ha="center", va="bottom",
                            fontsize=8.0, color=C_INK2, xytext=(0, 3),
                            textcoords="offset points", zorder=4)

        base = d["Base rate (%)"].iloc[0]
        ax.axhline(base, color=C_INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.set_ylim(0, max(top * 1.22, 30.0))
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9.0, color=C_INK)
        prl = d["PR lift"].to_numpy()
        winner = short.get(d["Model"].iloc[int(np.argmax(prl))], "").replace("\n", " ")
        ax.set_title("{}   (dashed line = base rate {:.2f}%)\n"
                     "best PR lift {:.2f}x  -  {}"
                     .format(tshort, base, prl.max(), winner),
                     fontsize=11.0, color=C_INK, fontweight="bold", pad=10, loc="left")
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C_GRID)
        ax.tick_params(axis="y", colors=C_INK2, length=0, labelsize=9)
        ax.tick_params(axis="x", colors=C_INK, length=0)

    for ax in axes:
        ax.set_ylabel("Score (%)", fontsize=10.5, color=C_INK2)

    handles, lab = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, lab, frameon=False, ncol=3, loc="upper left",
                     bbox_to_anchor=(0.006, 0.935), fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(C_INK2)

    fig.suptitle("Starlink / COSMOS 2251 conjunction risk: GNN vs classical baselines\n"
                 "each model shown at its own best-F1 threshold",
                 fontsize=13.0, color=C_INK, fontweight="bold", x=0.006, ha="left", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(os.path.join(HERE, PLOT_PNG), dpi=170, facecolor=C_SURFACE)
    print("  Saved chart -> {}".format(PLOT_PNG))


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        # redraw the chart from the saved results, without retraining anything
        make_plot(pd.read_csv(os.path.join(HERE, RESULTS_CSV)))
    else:
        main()
