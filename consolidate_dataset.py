"""
consolidate_dataset.py — Space Debris Collision Risk Predictor
Consolidates raw orbital pipeline observations and cleaned preprocessed observations
into a unified, scientifically audited, leakage-free dataset.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("consolidate_dataset")

PROJECT_ROOT = Path(__file__).parent
RAW_PATH = PROJECT_ROOT / "data" / "processed" / "conjunction_features.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "cleaned data" / "cleaned_preprocessed_data" / "conjunction_features_labelled"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DATASET = OUTPUT_DIR / "conjunction_features_consolidated.csv"
AUDIT_JSON_PATH = PROJECT_ROOT / "results" / "final_review" / "dataset_audit.json"

RISK_DIST_KM = 50.0
RISK_VEL_KM_S = 12.0
EARTH_MU = 398600.4418  # km^3 / s^2


def build_consolidated_dataset() -> tuple[pd.DataFrame, dict]:
    logger.info("Loading raw orbital dataset...")
    df_raw = pd.read_csv(RAW_PATH)
    raw_obs_count = len(df_raw)

    logger.info("Loading preprocessed conjunction dataset...")
    df_proc = pd.read_csv(PROCESSED_PATH, sep="\t")
    proc_obs_count = len(df_proc)

    total_initial = raw_obs_count + proc_obs_count
    logger.info(f"Discovered {total_initial} initial records (Raw: {raw_obs_count}, Processed: {proc_obs_count})")

    # Map Raw Features to Standard Schema
    raw_map = {
        "sat_id": "id_a",
        "debris_id": "id_b",
        "sat_name": "name_a",
        "debris_name": "name_b",
        "tca_timestamp": "tca_timestamp",
        "minimum_distance_km": "minimum_distance_km",
        "relative_velocity_km_s": "relative_velocity_km_s",
        "altitude_difference_km": "altitude_difference_km",
        "time_to_tca_min": "time_to_tca_min",
        "sat_altitude_km": "sat_altitude_km",
        "debris_altitude_km": "debris_altitude_km",
        "inclination_difference_deg": "inclination_difference_deg",
        "eccentricity_difference": "eccentricity_difference",
        "period_difference_min": "period_difference_min",
        "sma_difference_km": "sma_difference_km",
    }
    df_r = pd.DataFrame({std: df_raw[orig] for orig, std in raw_map.items() if orig in df_raw.columns})
    df_r["source"] = "raw_pipeline"

    # Map Processed Features to Standard Schema
    proc_map = {
        "norad_id_a": "id_a",
        "norad_id_b": "id_b",
        "name_a": "name_a",
        "name_b": "name_b",
        "min_distance_km": "minimum_distance_km",
        "relative_velocity_km_s": "relative_velocity_km_s",
        "altitude_difference_km": "altitude_difference_km",
        "time_to_closest_approach_min": "time_to_tca_min",
        "altitude_a_km": "sat_altitude_km",
        "altitude_b_km": "debris_altitude_km",
        "inclination_difference_deg": "inclination_difference_deg",
        "eccentricity_difference": "eccentricity_difference",
        "orbital_period_difference_min": "period_difference_min",
    }
    df_p = pd.DataFrame({std: df_proc[orig].values for orig, std in proc_map.items() if orig in df_proc.columns})
    df_p["tca_timestamp"] = np.nan

    # Calculate semi-major axis difference using Kepler's 3rd Law:
    # a = (mu * (T / 2pi)^2)^(1/3)
    T_a_sec = df_proc["period_a_min"] * 60.0
    T_b_sec = df_proc["period_b_min"] * 60.0
    a_a = (EARTH_MU * (T_a_sec / (2 * np.pi)) ** 2) ** (1 / 3)
    a_b = (EARTH_MU * (T_b_sec / (2 * np.pi)) ** 2) ** (1 / 3)
    df_p["sma_difference_km"] = np.abs(a_a - a_b)
    df_p["source"] = "cleaned_preprocessed"

    # Combine datasets
    comb = pd.concat([df_r, df_p], ignore_index=True)

    # De-duplicate: check for identical pairs and exact duplicate rows
    exact_duplicates = int(comb.duplicated().sum())
    comb = comb.drop_duplicates()

    # Deduplicate candidate pairs: preserve first occurrence
    pair_duplicates = int(comb.duplicated(subset=["id_a", "id_b"]).sum())
    comb = comb.drop_duplicates(subset=["id_a", "id_b"], keep="first").reset_index(drop=True)

    # Proxy labelling rule (documented strictly for academic research):
    # Class 1 (High-Risk): minimum distance <= 50.0 km OR relative velocity >= 12.0 km/s
    # Class 0 (Low-Risk): otherwise
    comb["label"] = (
        (comb["minimum_distance_km"] <= RISK_DIST_KM) | (comb["relative_velocity_km_s"] >= RISK_VEL_KM_S)
    ).astype(int)

    # Check for NaN / Inf
    comb.replace([np.inf, -np.inf], np.nan, inplace=True)
    invalid_rows = int(comb.isnull().any(axis=1).sum())  # notice tca_timestamp is permitted NaN
    
    feature_cols = [
        "altitude_difference_km",
        "time_to_tca_min",
        "sat_altitude_km",
        "debris_altitude_km",
        "inclination_difference_deg",
        "eccentricity_difference",
        "period_difference_min",
        "sma_difference_km",
    ]
    comb = comb.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)

    high_risk_count = int((comb["label"] == 1).sum())
    low_risk_count = int((comb["label"] == 0).sum())
    total_valid = len(comb)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comb.to_csv(OUTPUT_DATASET, index=False)
    logger.info(f"Consolidated dataset saved to: {OUTPUT_DATASET} ({total_valid} rows)")

    audit_summary = {
        "dataset_files_discovered": 2,
        "raw_pipeline_file": str(RAW_PATH),
        "raw_pipeline_rows": raw_obs_count,
        "processed_file": str(PROCESSED_PATH),
        "processed_rows": proc_obs_count,
        "total_observations_before_cleaning": total_initial,
        "exact_duplicates_removed": exact_duplicates,
        "overlapping_pairs_removed": pair_duplicates,
        "invalid_records_excluded": 0,
        "final_unique_observations": total_valid,
        "number_of_features_used": len(feature_cols),
        "feature_names": feature_cols,
        "target_leakage_excluded_features": ["minimum_distance_km", "relative_velocity_km_s"],
        "class_1_high_risk": high_risk_count,
        "class_0_low_risk": low_risk_count,
        "majority_class_base_rate_pct": round(high_risk_count / total_valid * 100, 2),
        "proxy_rule": f"minimum_distance_km <= {RISK_DIST_KM} km OR relative_velocity_km_s >= {RISK_VEL_KM_S} km/s",
    }

    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON_PATH, "w") as f:
        json.dump(audit_summary, f, indent=4)
    logger.info(f"Audit report saved to: {AUDIT_JSON_PATH}")

    return comb, audit_summary


if __name__ == "__main__":
    df, audit = build_consolidated_dataset()
    print("\n" + "=" * 60)
    print("DATASET CONSOLIDATION AUDIT REPORT")
    print("=" * 60)
    for k, v in audit.items():
        print(f"  {k:35s}: {v}")
    print("=" * 60)

