"""
propagation.py - SGP4 Orbital Propagation Engine.

Implements batched, discrete-time orbital propagation for celestial and artificial objects
using the SGP4 algorithm. Computes state vectors in the standard True Equator Mean Equinox
(TEME) Cartesian inertial reference frame.

Reference Frame Documentation:
------------------------------
- Position vectors (x, y, z): True Equator, Mean Equinox (TEME) frame in kilometers (km).
- Velocity vectors (vx, vy, vz): Inertial relative velocity in kilometers per second (km/s).
- SGP4 algorithm outputs native TEME coordinates; no coordinate system mixing is performed.
- Earth equatorial radius constant used for nominal altitude: WGS-84 / WGS-72 R_EARTH = 6378.137 km.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

from src.data_loader import OrbitalObject

logger = logging.getLogger("space_debris.propagation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

R_EARTH_KM: float = 6378.137


class SGP4Propagator:
    """
    Manages temporal discretization and SGP4 state vector propagation.
    """

    def __init__(
        self,
        prediction_window_minutes: float = 90.0,
        time_step_minutes: float = 2.0,
    ) -> None:
        self.prediction_window_minutes = float(prediction_window_minutes)
        self.time_step_minutes = float(time_step_minutes)

    def generate_timeline(
        self,
        start_time: datetime,
        window_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
    ) -> List[datetime]:
        """Generate discrete UTC timestamps spanning the simulation prediction window."""
        window = window_minutes if window_minutes is not None else self.prediction_window_minutes
        step = step_minutes if step_minutes is not None else self.time_step_minutes

        if step <= 0:
            raise ValueError(f"Time step must be positive, got {step}")

        total_steps = int(math_steps := np.ceil(window / step)) + 1
        timeline = [
            start_time + timedelta(minutes=i * step)
            for i in range(total_steps)
        ]
        return timeline

    def propagate_single(
        self,
        orbital_obj: OrbitalObject,
        timestamps: List[datetime],
    ) -> List[Dict[str, Any]]:
        """
        Propagate a single orbital object across the designated timestamps.
        Catches SGP4 internal numerical errors safely without aborting execution.
        """
        records: List[Dict[str, Any]] = []
        sat = orbital_obj.satrec

        for ts in timestamps:
            # Ensure timezone-aware UTC datetime
            ts_utc = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
            jd, fr = jday(
                ts_utc.year, ts_utc.month, ts_utc.day,
                ts_utc.hour, ts_utc.minute,
                ts_utc.second + ts_utc.microsecond / 1e6
            )

            error_code, r, v = sat.sgp4(jd, fr)

            if error_code == 0:
                # Valid state vector in TEME frame
                x, y, z = r
                vx, vy, vz = v
                radius = np.sqrt(x**2 + y**2 + z**2)
                altitude = radius - R_EARTH_KM
                is_valid = True
            else:
                # SGP4 error (e.g. decayed orbit, elements suborbital, etc.)
                x = y = z = np.nan
                vx = vy = vz = np.nan
                altitude = np.nan
                is_valid = False

            rec = {
                "norad_id": orbital_obj.norad_id,
                "object_name": orbital_obj.name,
                "object_type": orbital_obj.object_type,
                "timestamp": ts_utc,
                "epoch_minutes": (ts_utc - timestamps[0]).total_seconds() / 60.0,
                "x_km": x,
                "y_km": y,
                "z_km": z,
                "vx_km_s": vx,
                "vy_km_s": vy,
                "vz_km_s": vz,
                "altitude_km": altitude,
                "error_code": error_code,
                "is_valid": is_valid,
            }
            records.append(rec)

        return records

    def propagate_all(
        self,
        objects: List[OrbitalObject],
        start_time: Optional[datetime] = None,
        window_minutes: Optional[float] = None,
        step_minutes: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Propagate all objects across the prediction window.
        Returns a tidy pandas DataFrame containing time series states.
        """
        if not objects:
            logger.warning("No orbital objects provided for propagation.")
            return pd.DataFrame()

        # If start_time not provided, choose current UTC or median epoch
        if start_time is None:
            # Anchor propagation start time around the newest epoch to ensure realistic propagation
            epochs = [obj.epoch for obj in objects if obj.epoch is not None]
            start_time = max(epochs) if epochs else datetime.now(timezone.utc)

        start_time_utc = start_time if start_time.tzinfo is not None else start_time.replace(tzinfo=timezone.utc)
        timeline = self.generate_timeline(start_time_utc, window_minutes, step_minutes)

        window = window_minutes if window_minutes is not None else self.prediction_window_minutes
        step = step_minutes if step_minutes is not None else self.time_step_minutes

        logger.info(
            f"Propagating {len(objects)} objects over {window:.1f} min window "
            f"(step={step:.1f} min, {len(timeline)} time steps, starting {start_time_utc.isoformat()})..."
        )

        all_records: List[Dict[str, Any]] = []
        total_errors = 0

        for obj in objects:
            obj_records = self.propagate_single(obj, timeline)
            all_records.extend(obj_records)
            errors_in_obj = sum(1 for r in obj_records if not r["is_valid"])
            total_errors += errors_in_obj

        df_states = pd.DataFrame(all_records)
        valid_states = df_states["is_valid"].sum() if not df_states.empty else 0

        logger.info(
            f"Propagation finished: Total state samples={len(df_states)}, "
            f"Valid states={valid_states}, SGP4 Error samples={total_errors}"
        )
        return df_states

    @staticmethod
    def save_states(df: pd.DataFrame, filepath: Union[str, Path]) -> Path:
        """Save propagated states to parquet or csv."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} propagated state records to {path.resolve()}")
        return path

    @staticmethod
    def load_states(filepath: Union[str, Path]) -> pd.DataFrame:
        """Load propagated states from parquet or csv."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"State file not found: {path.resolve()}")
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)


if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    debris_objs = loader.load_csv("data/raw/cosmos_2251_debris.csv", object_type="debris")
    sat_objs = loader.load_csv("data/raw/starlink.csv", object_type="satellite")

    # Select sample objects for verification
    sample_objs = [debris_objs[0], sat_objs[0]]

    propagator = SGP4Propagator(prediction_window_minutes=60.0, time_step_minutes=10.0)
    df_sample = propagator.propagate_all(sample_objs)

    print("\n--- SAMPLE PROPAGATION OUTPUT (TEME Frame) ---")
    for obj in sample_objs:
        sub = df_sample[df_sample["norad_id"] == obj.norad_id].iloc[0]
        print(f"\nObject: {sub['object_name']} (NORAD: {sub['norad_id']}, Type: {sub['object_type']})")
        print(f"Timestamp: {sub['timestamp']}")
        print(f"Position (x, y, z) [km]: ({sub['x_km']:.3f}, {sub['y_km']:.3f}, {sub['z_km']:.3f})")
        print(f"Velocity (vx, vy, vz) [km/s]: ({sub['vx_km_s']:.3f}, {sub['vy_km_s']:.3f}, {sub['vz_km_s']:.3f})")
        print(f"Altitude [km]: {sub['altitude_km']:.2f}")
        print(f"SGP4 Error Code: {sub['error_code']} (Valid: {sub['is_valid']})")
