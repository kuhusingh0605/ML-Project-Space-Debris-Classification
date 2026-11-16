"""
main.py - End-to-End Space Debris Collision Risk Predictor Pipeline.

Executes all phases sequentially:
  Phase 1 : Load data
  Phase 2 : SGP4 propagation
  Phase 3 : Proximity filtering + Computational benchmarking
  Phase 4 : Feature engineering
  Phase 5 : Proxy labeling
  Phase 6 : ML model training
  Phase 7 : Model evaluation
  Phase 8 : Risk scoring
  Phase 9 : Save results

Usage:
  python main.py
  python main.py --config config/config.yaml
  python main.py --debris-csv data/raw/cosmos_2251_debris.csv --sat-csv data/raw/starlink.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# ---- Configure root logger ----
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("space_debris.main")

# ---- Project imports ----
from src.data_loader import DataLoader
from src.propagation import SGP4Propagator
from src.proximity_filter import ProximityFilter
from src.feature_engineering import FeatureEngineer
from src.labeling import ProxyLabeler
from src.models import ConjunctionDataset, ModelTrainer, FEATURE_COLUMNS
from src.evaluation import ModelEvaluator
from src.risk_scoring import RiskScorer
from src.preprocessing import validate_feature_dataframe, report_dataset_health


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        logger.warning(f"Config file not found at {path}. Using defaults.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(config: dict) -> None:
    """Execute the full analysis pipeline end-to-end."""
    t_pipeline_start = time.perf_counter()

    # ── Extract configuration values ───────────────────────────────────────
    data_cfg = config.get("data", {})
    prop_cfg = config.get("propagation", {})
    prox_cfg = config.get("proximity_filter", {})
    lbl_cfg = config.get("labeling", {})
    model_cfg = config.get("models", {})
    score_cfg = config.get("risk_scoring", {})
    seed = config.get("project", {}).get("random_seed", 42)

    debris_csv = data_cfg.get("debris_csv", "data/raw/cosmos_2251_debris.csv")
    satellites_csv = data_cfg.get("satellites_csv", "data/raw/starlink.csv")
    processed_features_path = data_cfg.get("processed_features", "data/processed/conjunction_features.csv")
    propagated_states_path = data_cfg.get("propagated_states", "data/processed/propagated_states.parquet")

    window_min = prop_cfg.get("prediction_window_minutes", 90.0)
    step_min = prop_cfg.get("time_step_minutes", 2.0)

    screen_threshold = prox_cfg.get("screening_threshold_km", 200.0)
    risk_dist_km = lbl_cfg.get("risk_distance_km", 50.0)

    lr_params = model_cfg.get("logistic_regression", {})
    rf_params = model_cfg.get("random_forest", {})
    test_size = model_cfg.get("test_size", 0.25)

    low_thresh = score_cfg.get("low_threshold", 0.3)
    high_thresh = score_cfg.get("high_threshold", 0.7)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1: DATA LOADING
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — DATA LOADING")
    print("="*60)

    loader = DataLoader()
    debris_objects = []
    satellite_objects = []

    for path, obj_type, store in [
        (debris_csv, "debris", "debris"),
        (satellites_csv, "satellite", "satellite"),
    ]:
        try:
            objs = loader.load_dataset(path, object_type=obj_type)
            if store == "debris":
                debris_objects = objs
            else:
                satellite_objects = objs
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            sys.exit(1)

    if not debris_objects or not satellite_objects:
        logger.error("No objects loaded. Aborting pipeline.")
        sys.exit(1)

    all_objects = debris_objects + satellite_objects
    print(f"\n  [OK] Debris objects loaded   : {len(debris_objects)}")
    print(f"  [OK] Satellite objects loaded : {len(satellite_objects)}")
    print(f"  [OK] Total objects            : {len(all_objects)}")
    print(f"\n  Sample Debris  : {debris_objects[0]}")
    print(f"  Sample Satellite: {satellite_objects[0]}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2: SGP4 PROPAGATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 — SGP4 ORBITAL PROPAGATION (TEME Frame)")
    print("="*60)

    propagator = SGP4Propagator(
        prediction_window_minutes=window_min,
        time_step_minutes=step_min,
    )
    df_states = propagator.propagate_all(all_objects)

    if df_states.empty:
        logger.error("Propagation produced no state vectors. Aborting.")
        sys.exit(1)

    valid_pct = 100.0 * df_states["is_valid"].sum() / len(df_states)
    print(f"\n  [OK] Total state samples      : {len(df_states):,}")
    print(f"  [OK] Valid SGP4 states        : {df_states['is_valid'].sum():,} ({valid_pct:.1f}%)")
    print(f"  [OK] SGP4 error samples       : {(~df_states['is_valid']).sum():,}")

    # Save propagated states
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    try:
        propagator.save_states(df_states, propagated_states_path)
    except Exception as e:
        logger.warning(f"Could not save propagated states: {e}")

    # Print sample state for verification
    sample_row = df_states[df_states["is_valid"]].iloc[0]
    print(f"\n  Sample State Vector:")
    print(f"    Object    : {sample_row['object_name']} (NORAD {sample_row['norad_id']})")
    print(f"    Timestamp : {sample_row['timestamp']}")
    print(f"    Position  : ({sample_row['x_km']:.2f}, {sample_row['y_km']:.2f}, {sample_row['z_km']:.2f}) km [TEME]")
    print(f"    Velocity  : ({sample_row['vx_km_s']:.3f}, {sample_row['vy_km_s']:.3f}, {sample_row['vz_km_s']:.3f}) km/s")
    print(f"    Altitude  : {sample_row['altitude_km']:.2f} km")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 3: PROXIMITY FILTERING + BENCHMARK
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 3 — LOCALITY PROXIMITY FILTERING")
    print(f"           (Experimental screening threshold: {screen_threshold:.1f} km)")
    print("="*60)

    proximity = ProximityFilter(screening_threshold_km=screen_threshold)
    candidates, benchmark = proximity.benchmark_screening(df_states)

    if not candidates:
        logger.warning(
            f"No candidate pairs found within {screen_threshold} km screening threshold. "
            "Try increasing screening_threshold_km in config.yaml."
        )
        sys.exit(1)

    print(f"\n  Computational Reduction Analysis:")
    print(f"  Total possible pairs       : {benchmark.total_possible_pairs:,}")
    print(f"  Pairs after locality filter: {benchmark.candidate_pairs_retained:,}")
    print(f"  Pairs eliminated           : {benchmark.pairs_removed:,}")
    print(f"  Locality reduction         : {benchmark.reduction_percentage:.2f}%")
    print(f"  Baseline (naive) runtime   : {benchmark.baseline_runtime_sec:.4f} s")
    print(f"  Filtered pipeline runtime  : {benchmark.filtered_runtime_sec:.4f} s")
    print(f"  Measured speedup factor    : {benchmark.speedup_factor:.2f}x")

    # Save benchmark
    benchmark_dict = {
        "total_possible_pairs": benchmark.total_possible_pairs,
        "candidate_pairs_retained": benchmark.candidate_pairs_retained,
        "pairs_removed": benchmark.pairs_removed,
        "reduction_percentage": round(benchmark.reduction_percentage, 4),
        "baseline_runtime_sec": round(benchmark.baseline_runtime_sec, 6),
        "filtered_runtime_sec": round(benchmark.filtered_runtime_sec, 6),
        "speedup_factor": round(benchmark.speedup_factor, 4),
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(benchmark_dict, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 4: FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 4 — FEATURE ENGINEERING")
    print("="*60)

    engineer = FeatureEngineer(
        sat_objects=satellite_objects,
        debris_objects=debris_objects,
    )
    df_features = engineer.build_features(candidates)

    if df_features.empty:
        logger.error("Feature extraction produced empty dataframe. Aborting.")
        sys.exit(1)

    print(f"\n  [OK] Feature matrix shape     : {df_features.shape}")
    print(f"  Features: {[c for c in FEATURE_COLUMNS if c in df_features.columns]}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 5: LABELING
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 5 — EXPERIMENTAL PROXY LABELING")
    print(f"           (Risk distance threshold: {risk_dist_km} km)")
    print("           NOTE: Experimental labels only — not official collision criteria")
    print("="*60)

    labeler = ProxyLabeler(risk_distance_km=risk_dist_km)
    df_labeled = labeler.apply_labels(df_features)

    # Validate labeled dataset
    is_valid, issues = validate_feature_dataframe(
        df_labeled, required_columns=FEATURE_COLUMNS[:5] + ["label"]
    )
    label_dist = labeler.label_distribution(df_labeled)
    print(f"\n  [OK] Label distribution: {label_dist}")

    if label_dist["class_1_high_priority"] == 0:
        logger.warning(
            "No Class=1 (High Priority) pairs found! "
            "Consider lowering risk_distance_km or widening screening_threshold_km. "
            "ML training will proceed but classification may be trivial."
        )

    # Save labeled features
    engineer.save_features(df_labeled, processed_features_path)
    report_dataset_health(df_labeled, name="Labeled Conjunction Features")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 6 & 7: ML TRAINING + EVALUATION
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 6 — ML MODEL TRAINING")
    print("="*60)

    dataset = ConjunctionDataset(random_state=seed)
    try:
        X_train, X_test, y_train, y_test = dataset.prepare(df_labeled, test_size=test_size)
    except ValueError as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

    print(f"\n  [OK] Training samples   : {len(X_train)}")
    print(f"  [OK] Test samples       : {len(X_test)}")
    print(f"  [OK] Features used      : {list(X_train.columns)}")

    trainer = ModelTrainer(random_state=seed)
    lr_train_params = {
        "C": lr_params.get("C", 1.0),
        "max_iter": lr_params.get("max_iter", 1000),
        "class_weight": lr_params.get("class_weight", "balanced"),
    }
    rf_train_params = {
        "n_estimators": rf_params.get("n_estimators", 100),
        "max_depth": rf_params.get("max_depth", 8),
        "class_weight": rf_params.get("class_weight", "balanced"),
    }
    trainer.train_all(X_train, y_train, lr_params=lr_train_params, rf_params=rf_train_params)

    # Save models
    Path("models").mkdir(parents=True, exist_ok=True)
    for model_name in trainer.trained_models:
        trainer.save_model(model_name)

    print("\n" + "="*60)
    print("  PHASE 7 — MODEL EVALUATION")
    print("="*60)

    evaluator = ModelEvaluator(results_dir="results/")
    model_results = {}

    for model_name in trainer.trained_models:
        y_pred, y_proba = trainer.predict(model_name, X_test)
        metrics = evaluator.evaluate(model_name, y_test.values, y_pred, y_proba)
        model_results[model_name] = (y_test.values, y_pred, y_proba)

        # Plots
        evaluator.plot_confusion_matrix(model_name, y_test.values, y_pred)

        feat_imp = trainer.get_feature_importances(model_name)
        if feat_imp is not None:
            evaluator.plot_feature_importance(model_name, feat_imp)

    evaluator.plot_roc_curves(model_results)
    evaluator.save_metrics_json()
    evaluator.print_comparison_table()

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 8 & 9: RISK SCORING + SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 8 — RISK SCORING & RANKING")
    print("="*60)

    # Use best model (RF by default; fallback to LR)
    primary_model = "Random Forest" if "Random Forest" in trainer.trained_models else list(trainer.trained_models.keys())[0]
    feature_cols = [c for c in FEATURE_COLUMNS if c in df_labeled.columns]
    df_for_scoring = df_labeled.dropna(subset=feature_cols)

    if df_for_scoring.empty:
        logger.warning("No rows for risk scoring after NaN removal.")
    else:
        X_all = df_for_scoring[feature_cols]
        y_pred_all, y_proba_all = trainer.predict(primary_model, X_all)

        scorer = RiskScorer(low_threshold=low_thresh, high_threshold=high_thresh)
        df_scored = scorer.score_conjunctions(df_for_scoring.copy(), y_pred_all, y_proba_all, model_name=primary_model)
        scorer.save_scored_results(df_scored, "results/risk_scored_conjunctions.csv")

        top_pairs = scorer.top_risky_pairs(df_scored, n=10)
        print(f"\n  Top Risky Conjunction Pairs (by model risk score):")
        if not top_pairs.empty:
            for _, row in top_pairs.iterrows():
                cat = row.get("risk_category", "N/A")
                print(
                    f"  [{cat}] {row.get('sat_name','?')} <-> {row.get('debris_name','?')}"
                    f" | Dist: {row.get('minimum_distance_km',0):.2f} km"
                    f" | Vel: {row.get('relative_velocity_km_s',0):.2f} km/s"
                    f" | Score: {row.get('risk_score',0):.4f}"
                )

    t_total = time.perf_counter() - t_pipeline_start
    print("\n" + "="*60)
    print(f"  PIPELINE COMPLETE [OK]  — Total runtime: {t_total:.2f} s")
    print(f"  Results saved to:    results/")
    print(f"  Models saved to:     models/")
    print(f"  Processed data:      data/processed/")
    print(f"\n  Launch dashboard:    streamlit run dashboard/app.py")
    print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Space Debris Collision Risk Predictor — Academic Feasibility Study"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument("--debris-csv", help="Override: path to debris CSV")
    parser.add_argument("--sat-csv", help="Override: path to satellite CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override: proximity screening threshold in km",
    )
    parser.add_argument(
        "--window",
        type=float,
        help="Override: prediction window in minutes",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply CLI overrides
    if args.debris_csv:
        config.setdefault("data", {})["debris_csv"] = args.debris_csv
    if args.sat_csv:
        config.setdefault("data", {})["satellites_csv"] = args.sat_csv
    if args.threshold:
        config.setdefault("proximity_filter", {})["screening_threshold_km"] = args.threshold
    if args.window:
        config.setdefault("propagation", {})["prediction_window_minutes"] = args.window

    run_pipeline(config)


if __name__ == "__main__":
    main()
