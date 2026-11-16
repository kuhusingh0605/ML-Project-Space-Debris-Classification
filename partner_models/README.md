# Collision Risk Assessment for Large Satellite Constellations

Graph neural network vs classical ML baselines, on collision labels derived from
real orbital mechanics.

Dataset: 4,000 Starlink satellites and 586 COSMOS 2251 debris objects, from
Space-Track TLEs.

---

## The problem with the first version of this project

The original `model.py` defined its risk label like this:

```python
orbital_delta = (abs(starlink_inclination - debris_inclination) * 10 +
                 abs(starlink_eccentricity - debris_eccentricity) * 500)
threshold = np.percentile(orbital_delta, 12.0)
df_pairs['is_high_risk'] = (orbital_delta <= threshold).astype(int)
```

The label is a deterministic formula over `inclination` and `eccentricity` — and
those same two columns were then handed to the model as input features. So the
models were not learning collision risk. They were reverse-engineering a
percentile threshold from the two numbers that defined it. That is why the SVM
reached 100% recall: the task was arithmetic, not prediction.

A GNN on that label would not fix anything. It would learn the same formula, and
probably *less* well than an RBF-SVM, because a graph model is not the right tool
for fitting a clean two-variable algebraic boundary.

So the label had to be replaced before any of the modelling was worth doing.

---

## Pipeline

### Stage A — `conjunction_labels.py`

Produces the ground truth. Nothing here involves machine learning.

1. **Parse real TLEs.** Element extraction is done by the `sgp4` library's
   `Satrec.twoline2rv`, not by hand-written column slicing, so there is no silent
   parsing error to chase.

2. **Propagate every object with SGP4** over a common 24-hour window anchored at
   the latest TLE epoch in the set, at 2-minute steps.

3. **Analytic apogee/perigee screening.** Two objects can only ever conjunct if
   their radial bands overlap. This algebraic test removes most pairs before any
   expensive computation — it is the step that makes the problem tractable, and it
   is the same O(n²) reduction the reference paper argues for.

4. **True minimum miss distance** for each surviving candidate pair.

5. **Refinement pass.** A 2-minute grid is far too coarse: at LEO relative
   velocities (~7.5 km/s) two minutes of flight is about 900 km, so a coarse
   minimum badly overestimates the real closest approach. The closest 60,000 pairs
   are re-propagated on a 5-second grid bracketing their coarse time of closest
   approach. Pairs are grouped by that time so each group only propagates the
   objects it actually involves, which keeps memory flat.

6. **MOID** — Minimum Orbital Intersection Distance. The closest the two *orbits*
   ever come, regardless of where each object sits along its path. Each orbit is
   sampled in true anomaly, converted to inertial coordinates, and all
   sample pairs are compared.

Output: `conjunction_cache.npz`.

### Stage B — `gnn_collision_model.py`

#### Why a graph, and not pairs

This was the central design question. A pairwise model treats every
(satellite, debris) pair as independent: row 47 knows nothing about row 48, even
when both involve the same debris object. A GNN's whole purpose is to remove that
independence — each object becomes a node that aggregates information from its
orbital neighbourhood, and a pair is scored from two neighbourhood-aware
embeddings rather than two isolated feature vectors.

So the pair does not disappear. **It becomes an edge to be classified.**

| | |
|---|---|
| **Nodes** | all 4,586 objects (Starlink *and* debris) in one graph |
| **Node features** | 13 — inclination, eccentricity, mean motion, B\*, semi-major axis, perigee/apogee altitude, and RAAN / argument of perigee / mean anomaly each split into sin and cos |
| **Message-passing edges** | k-nearest-neighbour graph (k=12) in orbital-regime space: semi-major axis, inclination, and RAAN as sin/cos |
| **Supervised edges** | the candidate conjunction pairs from Stage A |
| **Model** | 2-layer GraphSAGE encoder → MLP decoder on `[z_i, z_j, |z_i − z_j|, z_i · z_j]` |

Angles are encoded as sin/cos because RAAN, argument of perigee and mean anomaly
wrap at 360°. Fed in raw, a model treats 359° and 1° as maximally distant when
they are in fact adjacent.

The kNN graph is built from geometry alone and carries no label information, so
constructing it over all nodes leaks nothing into the test set.

#### Keeping the comparison honest

The baselines receive the **same information** in flat form —
`concat(node_i, node_j, |node_i − node_j|)`, 39 features. Every model shares one
stratified train/test split and one decision threshold (0.50). The only thing the
GNN has that the baselines do not is the graph structure. Without that control,
a GNN "winning" would prove nothing.

A fifth, non-learned entry is included: an **analytic shell-overlap heuristic**
that scores pairs on altitude-band overlap alone. If a trained model cannot beat
pure physics, it is not contributing.

---

## Two targets, deliberately

**Task 1 — MOID < 50 km.** The standard first-stage geometric screen in real
conjunction assessment. MOID is a smooth, deterministic function of both objects'
shape-and-orientation elements, so it is genuinely learnable — while still
requiring all ten values in a strongly non-linear combination, so no two-variable
shortcut reproduces it.

**Task 2 — 24-hour time-phased miss distance < 50 km.** The closest the two
*objects* actually come inside the window. This depends on relative phasing, which
is violently sensitive to small differences in mean motion.

Reporting Task 2 is the point, not an embarrassment. It shows where element-based
learning stops working, and it explains why operational systems propagate
trajectories instead of regressing on elements. A project that only reported
Task 1 would be hiding the more interesting result.

---

## Metrics

With this class imbalance, **ROC-AUC is misleading** — it looks flattering for
every model and hides the differences that matter. The headline numbers are:

- **PR-AUC** (average precision), which is sensitive to performance on the rare
  positive class.
- **PR lift** = PR-AUC ÷ base rate. `1.0x` means no better than random guessing.

Results are reported two ways:

1. **At a shared 0.50 cut** — needed for a strict like-for-like comparison, but it
   measures *calibration* as much as ranking quality. It penalises any model whose
   probabilities are not centred near 0.50.
2. **At each model's own best-F1 threshold** — the fair comparison of what each
   model can actually do, and what the chart shows. The optima differ a lot: the
   GNN's sits near 0.82, XGBoost's near 0.55.

This distinction matters. At the shared cut the RBF-SVM scores 0% recall, which
looks like total failure — but its PR-AUC beats the physics heuristic, so it
*ranks* pairs sensibly. Its Platt-scaled probabilities simply never cross 0.50.
Reporting only the fixed cut would have misrepresented it.

Training time is included because it is part of the scaling argument.

---

## Results

### Task 1 — MOID < 50 km (base rate 2.93%, test n = 24,000)

Each model at its own best-F1 threshold:

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | PR lift | Train |
|---|---|---|---|---|---|---|---|
| Analytic heuristic | 64.53 | 11.41 | 19.39 | 0.8719 | 0.1091 | 3.73× | 0.0s |
| Logistic Regression | 59.12 | 12.83 | 21.08 | 0.8840 | 0.1351 | 4.62× | 0.2s |
| SVM (RBF) | 39.32 | 12.04 | 18.43 | 0.8504 | 0.1107 | 3.78× | 3.7s |
| XGBoost | 45.44 | 14.24 | 21.69 | 0.8784 | 0.1363 | 4.66× | 2.2s |
| **GNN (GraphSAGE)** | 37.32 | **18.07** | **24.35** | **0.8949** | **0.1595** | **5.45×** | 86.8s |

The GNN wins on every threshold-free measure: PR-AUC 0.1595 against XGBoost's
0.1363, ROC-AUC 0.8949 against 0.8784, and the best F1 reachable at any threshold,
24.35 against 21.69. Note *how* it wins — not by flagging more pairs, but by being
the only model to push precision past 18% at a usable recall.

**This result depended on one architectural fix, and the failed version is worth
reporting.** The first implementation fed only the 64-dimensional GraphSAGE
embeddings to the decoder, and it **lost** to XGBoost: PR-AUC 0.1254 against
0.1363. Mean-aggregation smooths each node toward its orbital neighbours, which is
the point — but it also destroys the exact element values a geometric target like
MOID depends on. Adding a skip connection, so the decoder sees raw node features
*alongside* the learned embeddings, moved PR-AUC from 0.1254 to 0.1595 and
flipped the ranking.

The lesson generalises: the neighbourhood helps only when the model is not
*forced* to go through it.

### Task 2 — 24h time-phased miss distance < 50 km (base rate 0.55%)

| Model | Recall | Precision | F1 | PR-AUC | PR lift |
|---|---|---|---|---|---|
| Analytic heuristic | 62.41 | 2.27 | 4.38 | 0.0195 | 3.53× |
| Logistic Regression | 61.65 | 2.22 | 4.29 | 0.0197 | 3.56× |
| SVM (RBF) | 15.04 | 2.23 | 3.88 | 0.0143 | 2.59× |
| XGBoost | 33.08 | 2.22 | 4.15 | 0.0184 | 3.33× |
| GNN (GraphSAGE) | 69.17 | 2.06 | 4.00 | 0.0186 | 3.37× |

Every model lands between 2.6× and 3.6× PR lift, and the GNN (0.0186) is
indistinguishable from plain logistic regression (0.0197) — and from the
*untrained physics heuristic* (0.0195). **Nothing learns this target.**

That is the honest and more interesting finding. The time-phased miss distance
depends on where each object happens to be along its orbit at the moment of
closest approach, and that phasing is violently sensitive to small differences in
mean motion — two objects with near-identical elements can pass 10 km or 2,000 km
apart depending on timing. The information is not present in elements at epoch, so
no architecture recovers it. This is precisely why operational conjunction
screening *propagates trajectories* rather than regressing on element sets, and it
is worth stating rather than quietly dropping the task.

### Honest caveats on these numbers

- A PR lift of 5.45× is a real improvement over the base rate, but absolute
  precision stays at 18%. This is a **screening-triage** result — it narrows what
  needs expensive propagation — not a maneuver-decision result.
- **Single train/test split, single seed, no confidence intervals.** Small gaps
  between the middle entries (SVM / heuristic / logistic regression) should not be
  over-read. The GNN-vs-XGBoost gap on Task 1 is large enough to be meaningful;
  nothing in Task 2 is.
- Torch CPU runs are not bit-reproducible across threads, so PR-AUC moves by
  ~0.0005 between identical runs. Reported figures are from the final run.
- The GNN takes ~40× XGBoost's training time for this gain at this data size. The
  scaling argument says that ratio inverts as the constellation grows, but **this
  project does not demonstrate that** — it would need the 10⁵-object experiment.
- The analytic heuristic reaching 3.73× lift with zero training is a useful
  reference point: more than half of what the learned models achieve on Task 1
  comes from orbital geometry that needs no ML at all.

---

## Scope, stated honestly

The reference paper trained on synthetic constellations of up to 100,000
satellites using NVIDIA A100 GPUs (40 GB) over roughly 12 hours.

This project validates the approach at ~10⁴ objects on **real** Space-Track
elements, on CPU. It does **not** reproduce their 10⁵-object scaling experiment,
and does not claim to. At 4,586 nodes a GNN needs no GPU at all.

One finding supports the paper's scaling argument directly: the RBF-SVM could not
be trained on the full pair set, because kernel SVMs are O(n²) in the number of
samples. It had to be fitted on a 15,000-pair subsample, and this is flagged in the
results table. Classical kernel methods genuinely do not scale to
constellation-size screening.

---

## Running it

```bash
py -m pip install torch torch_geometric sgp4 xgboost scikit-learn pandas matplotlib

py conjunction_labels.py     # Stage A — physics labels  (writes conjunction_cache.npz)
py gnn_collision_model.py    # Stage B — models + comparison
```

Stage A must run first. Stage B re-reads the cache, so the risk thresholds in
`TASKS` can be changed and the models re-run without repeating the propagation.

Outputs:

| File | Contents |
|---|---|
| `conjunction_cache.npz` | node features, candidate pairs, miss distances, MOID |
| `results_gnn_vs_baselines.csv` | full metrics table, both tasks |
| `model_comparison.png` | grouped-bar comparison chart |

---

## Files

| File | Role |
|---|---|
| `conjunction_labels.py` | Stage A — SGP4 propagation, screening, MOID, labels |
| `gnn_collision_model.py` | Stage B — GNN + baselines + evaluation + chart |
| `model.py` | the original version, kept for comparison (circular label — see above) |

---

## Honest limitations

- **SGP4 only**, with no covariance. Real conjunction assessment uses covariance
  data and computes a probability of collision, not a distance threshold. Miss
  distance is a proxy.
- **One 24-hour window** from a single TLE epoch set. No epoch diversity, so there
  is no test of how the models behave as TLEs age.
- **MOID is sampled, not solved.** True-anomaly sampling gives an upper bound on
  MOID with resolution limited by the sample count; an analytic MOID solver would
  be tighter.
- **A fixed distance threshold is a simplification.** Operational screening volumes
  are ellipsoidal and depend on orbit uncertainty, not a single scalar radius.
- **The 50 km threshold is a modelling choice**, selected to leave enough positives
  to evaluate on. It is wider than an operational maneuver trigger.
