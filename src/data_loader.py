"""
data_loader.py - Orbital Data Ingestion and Normalization Module.

Supports loading and parsing of orbital datasets in both CSV (CelesTrak / Space-Track GP)
and TLE (2LE / 3LE) formats. Validates required orbital elements, removes duplicates,
handles malformed records gracefully, and constructs SGP4 Satrec objects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sgp4.api import Satrec, WGS72, jday

# Configure module logger
logger = logging.getLogger("space_debris.data_loader")
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
class OrbitalObject:
    """Represents a validated celestial object ready for SGP4 propagation."""
    norad_id: int
    name: str
    epoch: datetime
    object_type: str  # 'satellite' or 'debris'
    satrec: Satrec
    inclination_deg: float
    eccentricity: float
    mean_motion_rev_day: float
    raan_deg: float
    arg_pericenter_deg: float
    mean_anomaly_deg: float
    bstar: float
    raw_metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"OrbitalObject(NORAD={self.norad_id}, Name='{self.name}', "
            f"Type='{self.object_type}', Inc={self.inclination_deg:.2f}°, "
            f"Ecc={self.eccentricity:.5f}, MM={self.mean_motion_rev_day:.4f} rev/day)"
        )


class DataLoader:
    """
    Robust loader for orbital state records.
    Normalizes column nomenclature across CelesTrak and Space-Track schemas.
    """

    # Canonical column aliases mapping
    COLUMN_ALIASES: Dict[str, List[str]] = {
        "norad_id": ["norad_cat_id", "norad_id", "cat_id", "satnum", "catalog_number"],
        "name": ["object_name", "satname", "name", "satellite_name"],
        "epoch": ["epoch", "epoch_timestamp", "date"],
        "inclination": ["inclination", "inc", "incl"],
        "eccentricity": ["eccentricity", "ecc", "ecco"],
        "mean_motion": ["mean_motion", "mm", "no_kozai", "meanmotion"],
        "raan": ["ra_of_asc_node", "raan", "node", "right_ascension"],
        "arg_pericenter": ["arg_of_pericenter", "argp", "arg_peri", "argument_of_perigee"],
        "mean_anomaly": ["mean_anomaly", "ma", "mo", "meananomaly"],
        "bstar": ["bstar", "b_star", "bstar_drag"],
        "mean_motion_dot": ["mean_motion_dot", "n_dot", "ndot", "first_derivative_mean_motion"],
        "mean_motion_ddot": ["mean_motion_ddot", "n_ddot", "nddot", "second_derivative_mean_motion"],
    }

    def __init__(self, gravity_model: Any = WGS72) -> None:
        self.gravity_model = gravity_model

    def _normalize_column_name(self, col: str) -> str:
        """Map raw column name to canonical internal identifier."""
        clean_col = col.strip().lower().replace(" ", "_")
        for canonical, aliases in self.COLUMN_ALIASES.items():
            if clean_col == canonical or clean_col in aliases:
                return canonical
        return clean_col

    def load_csv(
        self,
        filepath: Union[str, Path],
        object_type: str = "debris",
    ) -> List[OrbitalObject]:
        """
        Load orbital records from a CSV file (e.g. CelesTrak GP format).
        Returns a list of validated OrbitalObject instances.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Orbital CSV file not found: {path.resolve()}")

        logger.info(f"Loading orbital CSV data from {path.name}...")
        df = pd.read_csv(path)
        total_records = len(df)

        # Normalize column headers
        col_rename = {col: self._normalize_column_name(col) for col in df.columns}
        df = df.rename(columns=col_rename)

        required_cols = [
            "norad_id", "name", "epoch", "inclination", "eccentricity",
            "mean_motion", "raan", "arg_pericenter", "mean_anomaly"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV {path.name} is missing mandatory orbital columns: {missing_cols}")

        valid_objects: List[OrbitalObject] = []
        invalid_count = 0
        seen_norad_ids = set()
        duplicate_count = 0

        xpdotp = 1440.0 / (2.0 * math.pi)
        deg2rad = math.pi / 180.0

        for idx, row in df.iterrows():
            try:
                # Basic parsing and validation
                raw_id = row["norad_id"]
                if pd.isna(raw_id):
                    invalid_count += 1
                    continue
                norad_id = int(float(raw_id))

                if norad_id in seen_norad_ids:
                    duplicate_count += 1
                    continue

                name = str(row["name"]).strip() if not pd.isna(row["name"]) else f"OBJ-{norad_id}"

                # Parse Epoch
                epoch_val = str(row["epoch"]).strip()
                epoch_dt = pd.to_datetime(epoch_val, utc=True).to_pydatetime()
                jd, fr = jday(
                    epoch_dt.year, epoch_dt.month, epoch_dt.day,
                    epoch_dt.hour, epoch_dt.minute,
                    epoch_dt.second + epoch_dt.microsecond / 1e6
                )
                jdsatepoch = jd + fr

                # Orbital elements
                inclo = float(row["inclination"]) * deg2rad
                ecco = float(row["eccentricity"])
                if ecco < 0 or ecco >= 1.0:
                    invalid_count += 1
                    continue

                no_kozai = float(row["mean_motion"]) / xpdotp
                nodeo = float(row["raan"]) * deg2rad
                argpo = float(row["arg_pericenter"]) * deg2rad
                mo = float(row["mean_anomaly"]) * deg2rad

                bstar = float(row.get("bstar", 0.0)) if not pd.isna(row.get("bstar", 0.0)) else 0.0
                raw_ndot = float(row.get("mean_motion_dot", 0.0)) if not pd.isna(row.get("mean_motion_dot", 0.0)) else 0.0
                raw_nddot = float(row.get("mean_motion_ddot", 0.0)) if not pd.isna(row.get("mean_motion_ddot", 0.0)) else 0.0

                ndot = raw_ndot / (xpdotp * 1440.0)
                nddot = raw_nddot / (xpdotp * 1440.0 * 1440.0)

                # Initialize SGP4 Satrec instance
                sat = Satrec()
                sat.sgp4init(
                    self.gravity_model,
                    "i",
                    norad_id,
                    jdsatepoch - 2433281.5,
                    bstar,
                    ndot,
                    nddot,
                    ecco,
                    argpo,
                    inclo,
                    mo,
                    no_kozai,
                    nodeo,
                )

                if sat.error != 0:
                    invalid_count += 1
                    continue

                obj = OrbitalObject(
                    norad_id=norad_id,
                    name=name,
                    epoch=epoch_dt,
                    object_type=object_type,
                    satrec=sat,
                    inclination_deg=float(row["inclination"]),
                    eccentricity=ecco,
                    mean_motion_rev_day=float(row["mean_motion"]),
                    raan_deg=float(row["raan"]),
                    arg_pericenter_deg=float(row["arg_pericenter"]),
                    mean_anomaly_deg=float(row["mean_anomaly"]),
                    bstar=bstar,
                    raw_metadata=row.to_dict(),
                )
                valid_objects.append(obj)
                seen_norad_ids.add(norad_id)

            except Exception as err:
                invalid_count += 1
                logger.debug(f"Row {idx} malformed: {err}")

        logger.info(
            f"CSV Summary for {path.name} -> Loaded: {total_records}, "
            f"Valid Retained: {len(valid_objects)}, "
            f"Duplicates Dropped: {duplicate_count}, Invalid/Malformed: {invalid_count}"
        )
        return valid_objects

    def load_tle(
        self,
        filepath: Union[str, Path],
        object_type: str = "debris",
    ) -> List[OrbitalObject]:
        """
        Load orbital records from a 2LE or 3LE text file.
        Returns a list of validated OrbitalObject instances.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Orbital TLE file not found: {path.resolve()}")

        logger.info(f"Loading orbital TLE data from {path.name}...")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        total_lines = len(raw_lines)
        valid_objects: List[OrbitalObject] = []
        invalid_count = 0
        duplicate_count = 0
        seen_norad_ids = set()

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            # Detect 3LE format (name followed by line 1 and line 2)
            if not line.startswith("1 ") and not line.startswith("2 "):
                obj_name = line
                if i + 2 < len(raw_lines) and raw_lines[i+1].startswith("1 ") and raw_lines[i+2].startswith("2 "):
                    line1 = raw_lines[i+1]
                    line2 = raw_lines[i+2]
                    i += 3
                else:
                    # Malformed record
                    invalid_count += 1
                    i += 1
                    continue
            elif line.startswith("1 "):
                obj_name = "UNKNOWN"
                if i + 1 < len(raw_lines) and raw_lines[i+1].startswith("2 "):
                    line1 = line
                    line2 = raw_lines[i+1]
                    i += 2
                else:
                    invalid_count += 1
                    i += 1
                    continue
            else:
                invalid_count += 1
                i += 1
                continue

            try:
                # Initialize SGP4 Satrec from two lines
                sat = Satrec.twoline2rv(line1, line2)
                if sat.error != 0:
                    invalid_count += 1
                    continue

                norad_id = int(line1[2:7].strip())
                if norad_id in seen_norad_ids:
                    duplicate_count += 1
                    continue

                # Extract human-readable orbital elements from Line 2
                inclination_deg = float(line2[8:16].strip())
                raan_deg = float(line2[17:25].strip())
                eccentricity = float("0." + line2[26:33].strip().replace(" ", "0"))
                arg_pericenter_deg = float(line2[34:42].strip())
                mean_anomaly_deg = float(line2[43:51].strip())
                mean_motion_rev_day = float(line2[52:63].strip())

                # Approximate epoch from sat.epoch
                epoch_dt = sat.epoch if hasattr(sat, "epoch") and sat.epoch else datetime.now(timezone.utc)

                obj = OrbitalObject(
                    norad_id=norad_id,
                    name=obj_name,
                    epoch=epoch_dt,
                    object_type=object_type,
                    satrec=sat,
                    inclination_deg=inclination_deg,
                    eccentricity=eccentricity,
                    mean_motion_rev_day=mean_motion_rev_day,
                    raan_deg=raan_deg,
                    arg_pericenter_deg=arg_pericenter_deg,
                    mean_anomaly_deg=mean_anomaly_deg,
                    bstar=sat.bstar,
                    raw_metadata={"line1": line1, "line2": line2},
                )
                valid_objects.append(obj)
                seen_norad_ids.add(norad_id)

            except Exception as err:
                invalid_count += 1
                logger.debug(f"TLE parse error for '{obj_name}': {err}")

        logger.info(
            f"TLE Summary for {path.name} -> Read Lines: {total_lines}, "
            f"Valid Retained: {len(valid_objects)}, "
            f"Duplicates Dropped: {duplicate_count}, Invalid/Malformed: {invalid_count}"
        )
        return valid_objects

    def load_dataset(
        self,
        filepath: Union[str, Path],
        object_type: str = "debris",
    ) -> List[OrbitalObject]:
        """Automatically detect file extension and load CSV or TLE format."""
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix in [".csv", ".txt"] and "tle" not in path.name.lower():
            try:
                return self.load_csv(path, object_type=object_type)
            except Exception as e:
                logger.warning(f"Failed CSV parse on {path.name}, attempting TLE parser: {e}")
                return self.load_tle(path, object_type=object_type)
        else:
            return self.load_tle(path, object_type=object_type)


if __name__ == "__main__":
    loader = DataLoader()
    print("--- Testing CSV Loading ---")
    debris_csv = loader.load_csv("data/raw/cosmos_2251_debris.csv", object_type="debris")
    sats_csv = loader.load_csv("data/raw/starlink.csv", object_type="satellite")
    print(f"Loaded {len(debris_csv)} debris objects from CSV.")
    print(f"Loaded {len(sats_csv)} satellite objects from CSV.")
    if debris_csv:
        print(f"Sample Debris: {debris_csv[0]}")
    if sats_csv:
        print(f"Sample Satellite: {sats_csv[0]}")

    print("\n--- Testing TLE Loading ---")
    debris_tle = loader.load_tle("data/raw/cosmos_2251_debris.tle", object_type="debris")
    sats_tle = loader.load_tle("data/raw/starlink.tle", object_type="satellite")
    print(f"Loaded {len(debris_tle)} debris objects from TLE.")
    print(f"Loaded {len(sats_tle)} satellite objects from TLE.")
