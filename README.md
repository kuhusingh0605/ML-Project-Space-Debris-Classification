# Space Debris Collision Risk Predictor

**B.Tech Machine Learning Project — Academic Feasibility Study**

| Team Member | Roll Number |
|-------------|-------------|
| Kuhu Singh | RA2411026030110 |
| Janhavi Sharma | RA2411026030114 |

---

## Overview

This project is an **academic feasibility study** investigating whether the **locality principle** can be applied at a small practical scale using:

- **SGP4** orbital propagation
- **Proximity-based broad-phase filtering** to reduce unnecessary pairwise comparisons
- **Classical Machine Learning** (Logistic Regression, Random Forest) to prioritize candidate conjunction pairs

**Inspired by:** Raj Shekhar Peri, *"Graph Neural Networks for Real-Time Collision Risk Assessment in Large Satellite Constellations"* (December 2025)

**Important Distinction:**
| Reference Paper | This Project |
|---|---|
| Large-scale GNN-based real-time collision risk assessment | Smaller feasibility study using classical ML |
| Graph Neural Networks | Logistic Regression + Random Forest |
| Designed for constellation-scale operations | Academic prototype for research validation |

**Research Question:**  
*"Can the locality principle be implemented at a smaller practical scale using SGP4, proximity filtering and classical ML to reduce unnecessary pairwise analysis?"*

---

## Academic Disclaimers

> **WARNING: This system does NOT compute official collision probabilities.**
>
> - Minimum separation values are **sample-based predicted estimates** from discrete-time SGP4 propagation.
> - Risk categories are **experimental model-priority screening labels**, not physical collision probabilities (P_c).
> - This system is **NOT operationally suitable** for real spacecraft collision avoidance decisions.
> - Proxy labels are rule-based and are clearly distinguished from real historical conjunction data.
> - The system is intended solely for academic research and educational purposes.

---

## Project Architecture

```
Space Debris Collision Risk Predictor
│
├── config/config.yaml          ← Central configuration (thresholds, windows, paths)
│
├── data/
│   ├── raw/
│   │   ├── cosmos_2251_debris.csv  ← COSMOS 2251 debris field (CelesTrak GP format)
│   │   ├── cosmos_2251_debris.tle  ← Same objects in TLE format
│   │   ├── starlink.csv            ← Starlink LEO constellation subset
│   │   └── starlink.tle            ← Same objects in TLE format
│   └── processed/
│       ├── propagated_states.*     ← Cached SGP4 state vectors
│       └── conjunction_features.csv ← ML feature matrix with proxy labels
│
├── src/
│   ├── data_loader.py          ← Phase 1: CSV/TLE parsing & normalization
│   ├── propagation.py          ← Phase 2: SGP4 propagation (TEME frame)
│   ├── proximity_filter.py     ← Phase 3: Locality screening + benchmarking
│   ├── feature_engineering.py  ← Phase 4: Orbital feature extraction
│   ├── labeling.py             ← Phase 5: Proxy risk labeling
│   ├── models.py               ← Phase 6: LR + RF training pipelines
│   ├── evaluation.py           ← Phase 7: Metrics + visualizations
│   ├── risk_scoring.py         ← Phase 8: Risk tier assignment
│   └── preprocessing.py        ← Data validation utilities
│
├── dashboard/app.py            ← Interactive Streamlit dashboard
├── models/                     ← Serialized trained models (.pkl)
├── results/                    ← Metrics JSON + plots (PNG)
├── main.py                     ← End-to-end pipeline runner
└── requirements.txt
```

---

## Methodology

### 1. Data Sources
- **COSMOS 2251 Debris**: Tracked debris field from the 2009 Iridium-COSMOS collision — one of the most well-known debris-generating events, making it scientifically relevant for studying debris interaction.
- **Starlink Satellites**: Active LEO constellation with orbital altitudes (~340–560 km) partially overlapping with COSMOS 2251 debris orbital band (~800 km).
- **Source Format**: CelesTrak GP (General Perturbations) CSV format with standard CCSDS columns.

### 2. SGP4 Propagation (TEME Frame)
SGP4 (Simplified General Perturbations 4) is the industry-standard algorithm for near-Earth object propagation using two-line element (TLE) data.

**Coordinate Frame**: TEME (True Equator, Mean Equinox) — an inertial Cartesian frame.  
**Units**: Position in km, Velocity in km/s.

```
Position vector (TEME): (x, y, z) in km
Velocity vector (TEME): (vx, vy, vz) in km/s
Altitude = |r| - R_EARTH  where R_EARTH = 6378.137 km
```

**Configuration:**
- Prediction window: 90 minutes (configurable)
- Time step: 2 minutes (configurable)
- Gravity model: WGS-72

### 3. Proximity Filtering (Locality Principle)
The core locality principle from the reference GNN paper: instead of computing distances for all N×M pairs, use spatial indexing (k-d tree) to identify only pairs that come within an experimental screening threshold.

```
Broad-phase screening threshold: 200 km (EXPERIMENTAL — not official)

Naive complexity:   O(N_sat × N_debris × T)
Filtered approach:  k-d tree queries at each time step → O(T × (N + M) log N)
```

**Computational reduction is measured experimentally** from real execution times (not fabricated).

### 4. Feature Engineering
For each candidate conjunction pair, the following features are extracted at the predicted closest approach:

| Feature | Description | Units |
|---------|-------------|-------|
| `minimum_distance_km` | Predicted minimum separation distance (sample-based) | km |
| `relative_velocity_km_s` | Relative speed at closest approach | km/s |
| `altitude_difference_km` | Difference in orbital altitudes at TCA | km |
| `time_to_tca_min` | Time from epoch to closest approach | min |
| `inclination_difference_deg` | Orbital plane angle difference | degrees |
| `eccentricity_difference` | Eccentricity disparity | dimensionless |
| `period_difference_min` | Orbital period difference | min |
| `sma_difference_km` | Semi-major axis difference | km |

### 5. Experimental Proxy Labeling
Since no historical conjunction database is used, labels are assigned via a transparent rule:

```
label = 1 (High Priority)  if minimum_distance_km <= 50 km
                                OR relative_velocity_km_s >= 12 km/s
label = 0 (Low Priority)   otherwise
```

**This threshold is configurable and experimentally chosen.** These labels serve as proxy targets for the ML classifier within the feasibility study scope. They are NOT validated against real conjunction assessment data.

### 6. Machine Learning

**Model 1 — Logistic Regression (Baseline)**
- LinearScaler + LogisticRegression pipeline
- Purpose: Establish a simple linear baseline
- Class weighting: balanced (addresses class imbalance)

**Model 2 — Random Forest Classifier (Primary)**
- StandardScaler + RandomForestClassifier pipeline
- Purpose: Capture non-linear orbital feature interactions
- Feature importance: Mean Decrease in Impurity (MDI)

**Group-Aware Train/Test Split:**  
To prevent data leakage, each unique (sat_id, debris_id) pair is treated as a group. The GroupShuffleSplit ensures that observations from the same pair do not appear in both train and test sets.

### 7. Risk Scoring
Model probability outputs are mapped to risk tiers:

```
score >= 0.70  →  HIGH    (Model-Priority — requires further analysis)
score >= 0.30  →  MEDIUM  (Monitoring-Priority)
score <  0.30  →  LOW     (Low Priority)
```

**Note:** HIGH does not mean a collision will occur. It identifies pairs the model assigns the highest screening priority for further detailed analysis.

---

## Installation

```bash
# 1. Clone or download the project
cd "ML Project"

# 2. Create a virtual environment (recommended)
python -m venv .venv

# 3. Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# 4. Install dependencies
python -m pip install -r requirements.txt

# 5. Verify SGP4 installation
python -c "from sgp4.api import Satrec; print('SGP4 ready')"
```

---

## Usage

### Run the Full Pipeline
```bash
python main.py
```

With custom configuration:
```bash
python main.py --config config/config.yaml
python main.py --threshold 300 --window 120
```

### Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Individual Modules
```bash
# Phase 1: Test data loading
python -m src.data_loader

# Phase 2: Test SGP4 propagation
python -m src.propagation

# Phase 3: Test proximity filtering
python -m src.proximity_filter

# Phase 4 & 5: Test feature extraction + labeling
python -m src.labeling
```

---

## Configuration Reference

Edit [`config/config.yaml`](config/config.yaml) to change any parameter:

```yaml
propagation:
  prediction_window_minutes: 90   # How long to propagate (minutes)
  time_step_minutes: 2.0          # Temporal resolution

proximity_filter:
  screening_threshold_km: 200.0   # Broad-phase screening distance

labeling:
  risk_distance_km: 50.0          # Primary proxy label threshold

risk_scoring:
  low_threshold: 0.3              # Score below this = LOW
  high_threshold: 0.7             # Score at or above this = HIGH
```

---

## Results Artifacts

### Final Integrated & Audited Results (results/final_review/)
Generated by running `python train_final.py`:

| File | Description |
|------|-------------|
| `results/final_review/final_metrics.csv` | Final quantitative metrics comparison across all models |
| `results/final_review/final_metrics.json` | Detailed JSON metrics report |
| `results/final_review/dataset_audit.json` | Complete dataset consolidation & sample audit log |
| `results/final_review/model_comparison_final.png` | Comprehensive multi-metric bar chart comparison |
| `results/final_review/roc_comparison_final.png` | ROC curve comparison with AUC scores |
| `results/final_review/precision_recall_comparison_final.png` | Precision-Recall curve comparison against base rate |
| `results/final_review/rf_confusion_matrix_final.png` | Confusion matrix for Improved Random Forest |
| `results/final_review/lr_confusion_matrix_final.png` | Confusion matrix for Improved Logistic Regression |
| `results/final_review/svm_confusion_matrix_final.png` | Confusion matrix for Partner Support Vector Machine |
| `results/final_review/feature_importance_final.png` | Gini impurity orbital feature importances |
| `results/final_review/dataset_class_distribution_final.png` | Class balance distribution across 217 valid events |
| `results/final_review/train_vs_test_overfitting_check_final.png` | Training vs test F1 comparison demonstrating generalization |
| `models/final/random_forest_improved.pkl` | Serialized Improved Random Forest classifier |
| `models/final/logistic_regression_improved.pkl` | Serialized Improved Logistic Regression pipeline |
| `models/final/svm_final.pkl` | Serialized Partner Support Vector Machine (SVC) |

---

## Final Project Integration & Methodology Update

### 1. Final Integrated Architecture
The final project integrates:
- **Primary Model**: Improved Random Forest (`random_forest_improved.py`) featuring compact RandomizedSearchCV and out-of-fold calibration.
- **Baseline Model**: Improved Logistic Regression (`logistic_regression_improved.py`) with StandardScaler pipeline.
- **Integrated Partner Model**: Support Vector Machine (SVC with RBF kernel and balanced class weights, preserved from partner repository).
- **Consolidated Dataset**: 217 genuine orbital conjunction observations (`data/processed/conjunction_features_consolidated.csv`).
- **Partner GNN**: Preserved without retraining under `partner_models/gnn_collision_model.py` and `partner_models/conjunction_cache.npz`.

### 2. Dataset Consolidation Methodology
- Integrated raw pipeline observations (28 rows) and preprocessed conjunction observations (189 rows).
- Standardized schema across both sources, mapped unique NORAD pair IDs, and removed duplicate pairs.
- Preserved genuine observations only (Zero synthetic data, Zero SMOTE, Zero fabricated conjunctions).
- Validated feature distributions and computed Keplerian semi-major axis differences.

### 3. Target Leakage Prevention
- **Eliminated direct proxy-defining variables**: `minimum_distance_km` and `relative_velocity_km_s` are strictly excluded from the feature matrix used for model training.
- **8 Pure Orbital Features Retained**:
  1. `eccentricity_difference`
  2. `sma_difference_km`
  3. `inclination_difference_deg`
  4. `altitude_difference_km`
  5. `period_difference_min`
  6. `sat_altitude_km`
  7. `time_to_tca_min`
  8. `debris_altitude_km`

### 4. Final Evaluation Results (Held-out Test Set, N=55)

| Model | Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | 89.09% | 89.09% | 100.00% | 94.23% | 50.00% | 0.5000 | 0.8909 |
| **Improved Random Forest (Best)** | **96.36%** | **100.00%** | **95.92%** | **97.92%** | **97.96%** | **0.9592** | **0.9955** |
| **Improved Logistic Regression** | 92.73% | 100.00% | 91.84% | 95.74% | 95.92% | 0.9864 | 0.9983 |
| **Partner Support Vector Machine (SVC)** | 94.55% | 100.00% | 93.88% | 96.84% | 96.94% | 0.9592 | 0.9954 |

*Note: In the presence of class imbalance (89.86% High-Risk in screening pre-filters), F1-Score and Balanced Accuracy are the scientifically valid metrics. The Improved Random Forest achieved the highest F1-score (97.92%) and Balanced Accuracy (97.96%) with zero false positives.*

### 5. Preserved Partner GNN Implementation
- Partner files are safely organized in `partner_models/`.
- SGP4 multi-object propagation cache (`conjunction_cache.npz`) containing 120,000 pair evaluations across 4,586 objects is preserved.
- Historical GNN GraphSAGE benchmark results are preserved in `partner_models/results_gnn_vs_baselines.csv` and documented without retraining.

---

## Execution Instructions

To execute the complete end-to-end integrated training and evaluation pipeline:
```bash
python train_final.py
```

To run dataset consolidation independently:
```bash
python consolidate_dataset.py
```

To run the interactive dashboard:
```bash
streamlit run dashboard/app.py
```

---

## Limitations and Future Work

### Current Limitations
1. **Discrete-time TCA estimation**: Minimum separation is the minimum across sampled time steps, not an analytically interpolated TCA. Results may miss closer approaches between time steps.
2. **Proxy labels**: No real historical conjunction data (CDM records) used. Labels are experimental.
3. **Dataset scope**: Analysis limited to 250 Starlink satellites and ~222 COSMOS 2251 debris objects.
4. **No coordinate frame conversion**: TEME frame is used throughout without ECI/ECEF conversion for altitude calculations (negligible difference for this scale).

### Potential Improvements
- Integrate real CDM (Conjunction Data Message) records from Space-Track
- Add analytical TCA interpolation (parabolic fit on sampled distances)
- Implement ECEF/ECI frame transformations for precise ground-track analysis
- Scale to full Starlink constellation (~7,000 objects)
- Replace classical ML with the GNN architecture from the reference paper

---

## References

1. Peri, R.S. (December 2025). *"Graph Neural Networks for Real-Time Collision Risk Assessment in Large Satellite Constellations"*. (Inspiration for locality principle)
2. Hoots, F.R. & Roehrich, R.L. (1980). *"Models for Propagation of NORAD Element Sets"*. Spacetrack Report No. 3.
3. Vallado, D.A. et al. (2006). *"Revisiting Spacetrack Report #3"*. AIAA 2006-6753.
4. CelesTrak General Perturbations (GP) data: https://celestrak.org
5. SGP4 Python library: https://pypi.org/project/sgp4/

---

*This project is submitted for academic evaluation. Results are from real orbital data and real ML computations. No data, metrics, or scientific results have been fabricated.*
