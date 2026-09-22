"""
proximity_filter.py - Broad-Phase Locality Filtering & Closest Approach (TCA) Module.

Implements broad-phase spatial proximity filtering between operational satellites
and space debris. Replaces naive all-to-all pair brute-force search with locality-based
filtering to discard spatially distant pairs early.

Academic Note:
--------------
The screening distance threshold is an experimental screening filter designed for
computational triage. It is NOT an official collision avoidance or conjunction threshold.
Minimum distance estimates are sample-based predicted minimum separations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger("space_debris.proximity_filter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class CandidateConjunction:
    """Represents a screened candidate pair at its predicted closest approach."""
    sat_id: int
    sat_name: str
    debris_id: int
    debris_name: str
    tca_timestamp: datetime
    time_to_tca_min: float
    min_separation_km: float
    relative_velocity_km_s: float
    sat_altitude_km: float
    debris_altitude_km: float
    altitude_diff_km: float
    sat_pos_km: Tuple[float, float, float]
    deb_pos_km: Tuple[float, float, float]
    sat_vel_km_s: Tuple[float, float, float]
    deb_vel_km_s: Tuple[float, float, float]


@dataclass
class ComputationalBenchmark:
    """Benchmark metrics comparing naive vs proximity-filtered screening."""
    total_possible_pairs: int
    candidate_pairs_retained: int
    pairs_removed: int
    reduction_percentage: float
    baseline_runtime_sec: float
    filtered_runtime_sec: float
    speedup_factor: float


class ProximityFilter:
    """
    Broad-phase proximity filter and conjunction analyzer.
    """

    def __init__(self, screening_threshold_km: float = 200.0) -> None:
        self.screening_threshold_km = float(screening_threshold_km)

    def filter_conjunctions(
        self,
        df_states: pd.DataFrame,
        screening_threshold_km: Optional[float] = None,
    ) -> List[CandidateConjunction]:
        """
        Screen state vectors across timestamps for satellite-debris proximity.
        Uses k-d trees at each time slice to achieve sub-quadratic spatial queries.
        Deduplicates pairs: A-B == B-A.
        """
        threshold = screening_threshold_km if screening_threshold_km is not None else self.screening_threshold_km
        if df_states.empty:
            logger.warning("Empty states dataframe passed to proximity filter.")
            return []

        # Filter out invalid SGP4 states
        valid_df = df_states[df_states["is_valid"]].copy()
        if valid_df.empty:
            logger.warning("No valid SGP4 states found for proximity filtering.")
            return []

        # Separate satellites and debris
        sat_mask = valid_df["object_type"] == "satellite"
        deb_mask = valid_df["object_type"] == "debris"

        sat_df = valid_df[sat_mask]
        deb_df = valid_df[deb_mask]

        if sat_df.empty or deb_df.empty:
            logger.warning(
                f"Missing object groups: satellites={len(sat_df)}, debris={len(deb_df)}. "
                "Conjunctions require both satellites and debris."
            )
            return []

        unique_timestamps = sorted(valid_df["timestamp"].unique())
        logger.info(
            f"Screening conjunctions across {len(unique_timestamps)} time steps "
            f"with threshold={threshold:.1f} km..."
        )

        # Track pairwise distance time series for pairs that breach the screening threshold:
        # Key: (sat_id, debris_id) -> list of sample observations
        candidate_pair_samples: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

        t0 = unique_timestamps[0]

        for ts in unique_timestamps:
            s_slice = sat_df[sat_df["timestamp"] == ts]
            d_slice = deb_df[deb_df["timestamp"] == ts]

            if s_slice.empty or d_slice.empty:
                continue

            # Extract 3D Cartesian coordinates
            s_pos = s_slice[["x_km", "y_km", "z_km"]].values
            d_pos = d_slice[["x_km", "y_km", "z_km"]].values

            # Build k-d tree on debris positions
            tree = cKDTree(d_pos)
            # Query all debris within threshold of any satellite at this timestamp
            matched_indices = tree.query_ball_point(s_pos, r=threshold)

            for s_idx, d_matches in enumerate(matched_indices):
                if not d_matches:
                    continue
                s_row = s_slice.iloc[s_idx]
                sat_id = int(s_row["norad_id"])

                for d_match_idx in d_matches:
                    d_row = d_slice.iloc[d_match_idx]
                    deb_id = int(d_row["norad_id"])
                    pair_key = (sat_id, deb_id)

                    # Compute exact separation distance
                    dx = s_row["x_km"] - d_row["x_km"]
                    dy = s_row["y_km"] - d_row["y_km"]
                    dz = s_row["z_km"] - d_row["z_km"]
                    dist = float(np.sqrt(dx**2 + dy**2 + dz**2))

                    # Compute relative velocity vector and magnitude
                    dvx = s_row["vx_km_s"] - d_row["vx_km_s"]
                    dvy = s_row["vy_km_s"] - d_row["vy_km_s"]
                    dvz = s_row["vz_km_s"] - d_row["vz_km_s"]
                    rel_vel = float(np.sqrt(dvx**2 + dvy**2 + dvz**2))

                    sample = {
                        "timestamp": ts,
                        "time_to_tca_min": (ts - t0).total_seconds() / 60.0,
                        "distance_km": dist,
                        "relative_velocity_km_s": rel_vel,
                        "sat_name": s_row["object_name"],
                        "debris_name": d_row["object_name"],
                        "sat_altitude_km": s_row["altitude_km"],
                        "debris_altitude_km": d_row["altitude_km"],
                        "altitude_diff_km": abs(s_row["altitude_km"] - d_row["altitude_km"]),
                        "sat_pos": (s_row["x_km"], s_row["y_km"], s_row["z_km"]),
                        "deb_pos": (d_row["x_km"], d_row["y_km"], d_row["z_km"]),
                        "sat_vel": (s_row["vx_km_s"], s_row["vy_km_s"], s_row["vz_km_s"]),
                        "deb_vel": (d_row["vx_km_s"], d_row["vy_km_s"], d_row["vz_km_s"]),
                    }

                    if pair_key not in candidate_pair_samples:
                        candidate_pair_samples[pair_key] = []
                    candidate_pair_samples[pair_key].append(sample)

        # For each candidate pair, determine the predicted minimum separation and TCA
        candidates: List[CandidateConjunction] = []
        for (sat_id, deb_id), samples in candidate_pair_samples.items():
            # Find the sample with minimum separation distance
            closest_sample = min(samples, key=lambda s: s["distance_km"])

            candidate = CandidateConjunction(
                sat_id=sat_id,
                sat_name=closest_sample["sat_name"],
                debris_id=deb_id,
                debris_name=closest_sample["debris_name"],
                tca_timestamp=closest_sample["timestamp"],
                time_to_tca_min=closest_sample["time_to_tca_min"],
                min_separation_km=closest_sample["distance_km"],
                relative_velocity_km_s=closest_sample["relative_velocity_km_s"],
                sat_altitude_km=closest_sample["sat_altitude_km"],
                debris_altitude_km=closest_sample["debris_altitude_km"],
                altitude_diff_km=closest_sample["altitude_diff_km"],
                sat_pos_km=closest_sample["sat_pos"],
                deb_pos_km=closest_sample["deb_pos"],
                sat_vel_km_s=closest_sample["sat_vel"],
                deb_vel_km_s=closest_sample["deb_vel"],
            )
            candidates.append(candidate)

        # Sort candidates ascending by minimum separation distance
        candidates.sort(key=lambda c: c.min_separation_km)
        logger.info(f"Retained {len(candidates)} candidate conjunction pairs below {threshold:.1f} km.")
        return candidates

    def benchmark_screening(
        self,
        df_states: pd.DataFrame,
        screening_threshold_km: Optional[float] = None,
    ) -> Tuple[List[CandidateConjunction], ComputationalBenchmark]:
        """
        Directly measures and benchmarks Naive all-pairs evaluation vs. Filtered screening.
        Calculates empirical execution times and reduction metrics without fabricating numbers.
        """
        threshold = screening_threshold_km if screening_threshold_km is not None else self.screening_threshold_km
        valid_df = df_states[df_states["is_valid"]].copy()

        sat_df = valid_df[valid_df["object_type"] == "satellite"]
        deb_df = valid_df[valid_df["object_type"] == "debris"]

        n_sats = sat_df["norad_id"].nunique()
        n_debs = deb_df["norad_id"].nunique()
        total_possible_pairs = n_sats * n_debs

        unique_timestamps = sorted(valid_df["timestamp"].unique())

        # 1. BASELINE: Naive pairwise calculation (simulated across time samples)
        # To avoid freeze on very large datasets, we evaluate across time steps directly
        logger.info("Executing Baseline (Naive pairwise screening)...")
        t_start_baseline = time.perf_counter()

        naive_closest_dist: Dict[Tuple[int, int], float] = {}
        for ts in unique_timestamps:
            s_pts = sat_df[sat_df["timestamp"] == ts][["norad_id", "x_km", "y_km", "z_km"]].values
            d_pts = deb_df[deb_df["timestamp"] == ts][["norad_id", "x_km", "y_km", "z_km"]].values

            if len(s_pts) == 0 or len(d_pts) == 0:
                continue

            # Full matrix subtraction
            s_coords = s_pts[:, 1:4]
            d_coords = d_pts[:, 1:4]

            # Broadcast pairwise Euclidean distance: shape (n_sat, n_deb)
            diff = s_coords[:, np.newaxis, :] - d_coords[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))

            # Update minimums
            for i, s_row in enumerate(s_pts):
                s_id = int(s_row[0])
                for j, d_row in enumerate(d_pts):
                    d_id = int(d_row[0])
                    pair = (s_id, d_id)
                    cur_d = dists[i, j]
                    if pair not in naive_closest_dist or cur_d < naive_closest_dist[pair]:
                        naive_closest_dist[pair] = cur_d

        t_end_baseline = time.perf_counter()
        baseline_runtime = t_end_baseline - t_start_baseline

        # 2. PROPOSED: Locality-filtered screening using spatial indexing
        logger.info("Executing Proposed Pipeline (Locality-filtered screening)...")
        t_start_filtered = time.perf_counter()
        candidates = self.filter_conjunctions(df_states, threshold)
        t_end_filtered = time.perf_counter()
        filtered_runtime = t_end_filtered - t_start_filtered

        candidate_count = len(candidates)
        pairs_removed = total_possible_pairs - candidate_count
        reduction_pct = (pairs_removed / total_possible_pairs * 100.0) if total_possible_pairs > 0 else 0.0
        speedup = (baseline_runtime / filtered_runtime) if filtered_runtime > 0 else 1.0

        benchmark = ComputationalBenchmark(
            total_possible_pairs=total_possible_pairs,
            candidate_pairs_retained=candidate_count,
            pairs_removed=pairs_removed,
            reduction_percentage=reduction_pct,
            baseline_runtime_sec=baseline_runtime,
            filtered_runtime_sec=filtered_runtime,
            speedup_factor=speedup,
        )

        logger.info(
            f"Computational Benchmark Results:\n"
            f"  - Total possible pairs: {benchmark.total_possible_pairs:,}\n"
            f"  - Candidate pairs retained: {benchmark.candidate_pairs_retained:,}\n"
            f"  - Pairs removed: {benchmark.pairs_removed:,}\n"
            f"  - Locality reduction: {benchmark.reduction_percentage:.2f}%\n"
            f"  - Baseline runtime: {benchmark.baseline_runtime_sec:.3f} s\n"
            f"  - Filtered runtime: {benchmark.filtered_runtime_sec:.3f} s\n"
            f"  - Measured speedup: {benchmark.speedup_factor:.2f}x"
        )
        return candidates, benchmark


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.propagation import SGP4Propagator

    loader = DataLoader()
    deb_objs = loader.load_csv("data/raw/cosmos_2251_debris.csv", object_type="debris")[:50]
    sat_objs = loader.load_csv("data/raw/starlink.csv", object_type="satellite")[:50]

    propagator = SGP4Propagator(prediction_window_minutes=60.0, time_step_minutes=5.0)
    df_states = propagator.propagate_all(deb_objs + sat_objs)

    proximity = ProximityFilter(screening_threshold_km=500.0)
    candidates, benchmark = proximity.benchmark_screening(df_states)

    print("\n--- SAMPLE CANDIDATE CONJUNCTIONS ---")
    for c in candidates[:5]:
        print(
            f"Sat: {c.sat_name} (ID: {c.sat_id}) <---> Debris: {c.debris_name} (ID: {c.debris_id})\n"
            f"  TCA: {c.tca_timestamp} (+{c.time_to_tca_min:.1f} min)\n"
            f"  Min Separation: {c.min_separation_km:.2f} km\n"
            f"  Rel Velocity: {c.relative_velocity_km_s:.3f} km/s\n"
            f"  Alt Diff: {c.altitude_diff_km:.2f} km\n"
        )
