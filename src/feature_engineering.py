"""
feature_engineering.py - Orbital Conjunction Feature Extraction Module.

Generates a tabular feature dataset from candidate conjunction pairs produced
by the proximity filter. Each row represents one satellite-debris candidate pair
at its predicted closest approach (TCA), described by physically meaningful features
extracted from orbital mechanics.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.proximity_filter import CandidateConjunction
from src.data_loader import OrbitalObject

logger = logging.getLogger("space_debris.feature_engineering")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Earth gravitational parameter (mu) in km^3/s^2
MU_EARTH_KM3_S2: float = 398600.4418
# Earth equatorial radius in km
R_EARTH_KM: float = 6378.137


def _orbital_period_minutes(mean_motion_rev_day: float) -> float:
    """Compute orbital period in minutes from mean motion (rev/day)."""
    if mean_motion_rev_day <= 0:
        return float("nan")
    return 1440.0 / mean_motion_rev_day


def _semimajor_axis_km(mean_motion_rev_day: float) -> float:
    """Compute semi-major axis in km from mean motion (rev/day) using Kepler's 3rd law."""
    if mean_motion_rev_day <= 0:
        return float("nan")
    n_rad_s = (mean_motion_rev_day * 2.0 * math.pi) / 86400.0
    return (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)


class FeatureEngineer:
    """
    Constructs a machine-learning ready feature matrix from candidate conjunction pairs.
    Enriches conjunction data with orbital element differences and other geometric features.
    """

    def __init__(
        self,
        sat_objects: Optional[List[OrbitalObject]] = None,
        debris_objects: Optional[List[OrbitalObject]] = None,
    ) -> None:
        """
        Optionally provide orbital object lookup maps for orbital element features.
        """
        self._sat_map = {obj.norad_id: obj for obj in (sat_objects or [])}
        self._deb_map = {obj.norad_id: obj for obj in (debris_objects or [])}

    def build_features(
        self,
        candidates: List[CandidateConjunction],
    ) -> pd.DataFrame:
        """
        Convert candidate conjunction pairs into a structured ML feature table.
        Returns a pandas DataFrame where each row is one candidate pair.
        """
        if not candidates:
            logger.warning("No candidate conjunctions provided; returning empty feature table.")
            return pd.DataFrame()

        logger.info(f"Extracting features for {len(candidates)} candidate pairs...")

        rows = []
        for conj in candidates:
            sat_obj = self._sat_map.get(conj.sat_id)
            deb_obj = self._deb_map.get(conj.debris_id)

            # ---- Core proximity features ----
            min_dist_km = conj.min_separation_km
            rel_vel_km_s = conj.relative_velocity_km_s
            alt_diff_km = conj.altitude_diff_km
            time_to_tca_min = conj.time_to_tca_min

            # ---- Orbital element difference features ----
            incl_diff_deg = float("nan")
            ecc_diff = float("nan")
            period_diff_min = float("nan")
            sma_diff_km = float("nan")
            sat_incl_deg = float("nan")
            deb_incl_deg = float("nan")
            sat_ecc = float("nan")
            deb_ecc = float("nan")
            sat_period_min = float("nan")
            deb_period_min = float("nan")
            sat_sma_km = float("nan")
            deb_sma_km = float("nan")

            if sat_obj is not None:
                sat_incl_deg = sat_obj.inclination_deg
                sat_ecc = sat_obj.eccentricity
                sat_period_min = _orbital_period_minutes(sat_obj.mean_motion_rev_day)
                sat_sma_km = _semimajor_axis_km(sat_obj.mean_motion_rev_day)

            if deb_obj is not None:
                deb_incl_deg = deb_obj.inclination_deg
                deb_ecc = deb_obj.eccentricity
                deb_period_min = _orbital_period_minutes(deb_obj.mean_motion_rev_day)
                deb_sma_km = _semimajor_axis_km(deb_obj.mean_motion_rev_day)

            if sat_obj is not None and deb_obj is not None:
                # Signed and absolute differences
                incl_diff_deg = abs(sat_obj.inclination_deg - deb_obj.inclination_deg)
                ecc_diff = abs(sat_obj.eccentricity - deb_obj.eccentricity)
                period_diff_min = abs(sat_period_min - deb_period_min)
                sma_diff_km = abs(sat_sma_km - deb_sma_km)

            row = {
                # === Identifiers (not features for ML) ===
                "sat_id": conj.sat_id,
                "sat_name": conj.sat_name,
                "debris_id": conj.debris_id,
                "debris_name": conj.debris_name,
                "tca_timestamp": conj.tca_timestamp,

                # === FEATURE 1: Minimum predicted separation (km) ===
                # Most direct indicator of proximity risk
                "minimum_distance_km": min_dist_km,

                # === FEATURE 2: Relative velocity at closest approach (km/s) ===
                # Higher relative velocity → greater potential impact energy
                "relative_velocity_km_s": rel_vel_km_s,

                # === FEATURE 3: Altitude difference (km) ===
                # Objects at similar altitudes have higher sustained proximity likelihood
                "altitude_difference_km": alt_diff_km,

                # === FEATURE 4: Time to closest approach (minutes from sim start) ===
                "time_to_tca_min": time_to_tca_min,

                # === FEATURE 5 & 6: Individual altitudes (km) ===
                "sat_altitude_km": conj.sat_altitude_km,
                "debris_altitude_km": conj.debris_altitude_km,

                # === FEATURE 7: Inclination difference (degrees) ===
                # Orbital plane crossing angle — large values imply fast crossing
                "inclination_difference_deg": incl_diff_deg,

                # === FEATURE 8: Eccentricity difference ===
                # Dissimilar eccentricities can bring objects into shared orbital altitude bands
                "eccentricity_difference": ecc_diff,

                # === FEATURE 9 & 10: Orbital period difference and SMA difference ===
                "period_difference_min": period_diff_min,
                "sma_difference_km": sma_diff_km,

                # === Individual orbital element values (for analysis) ===
                "sat_inclination_deg": sat_incl_deg,
                "debris_inclination_deg": deb_incl_deg,
                "sat_eccentricity": sat_ecc,
                "debris_eccentricity": deb_ecc,
                "sat_period_min": sat_period_min,
                "debris_period_min": deb_period_min,
                "sat_sma_km": sat_sma_km,
                "debris_sma_km": deb_sma_km,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Report feature completeness
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_nan = df[numeric_cols].isna().sum().sum()
        total_cells = len(df) * len(numeric_cols)
        logger.info(
            f"Feature matrix built: {len(df)} pairs × {len(df.columns)} columns | "
            f"NaN cells: {n_nan}/{total_cells} "
            f"({'%.1f' % (n_nan/total_cells*100 if total_cells > 0 else 0)}%) — "
            "NaNs indicate missing orbital element lookups."
        )

        return df

    def save_features(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path] = "data/processed/conjunction_features.csv",
    ) -> Path:
        """Save the feature dataframe to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Feature dataset saved to {path.resolve()} ({len(df)} rows, {len(df.columns)} columns)")
        return path


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.propagation import SGP4Propagator
    from src.proximity_filter import ProximityFilter

    loader = DataLoader()
    deb_objs = loader.load_csv("data/raw/cosmos_2251_debris.csv", object_type="debris")
    sat_objs = loader.load_csv("data/raw/starlink.csv", object_type="satellite")

    propagator = SGP4Propagator(prediction_window_minutes=90.0, time_step_minutes=2.0)
    df_states = propagator.propagate_all(deb_objs + sat_objs)

    proximity = ProximityFilter(screening_threshold_km=200.0)
    candidates = proximity.filter_conjunctions(df_states)

    engineer = FeatureEngineer(sat_objects=sat_objs, debris_objects=deb_objs)
    df_features = engineer.build_features(candidates)
    engineer.save_features(df_features)

    print(f"\n--- FEATURE MATRIX SUMMARY ---")
    print(f"Shape: {df_features.shape}")
    print(f"\nColumn list:")
    for col in df_features.columns:
        print(f"  {col}")
    print(f"\nFirst 3 rows (key features):")
    key_cols = ["sat_name", "debris_name", "minimum_distance_km", "relative_velocity_km_s",
                "altitude_difference_km", "inclination_difference_deg", "eccentricity_difference"]
    if not df_features.empty:
        print(df_features[key_cols].head(3).to_string(index=False))
