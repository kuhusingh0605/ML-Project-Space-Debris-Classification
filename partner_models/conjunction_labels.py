"""
=======================================================================================
 STAGE A - PHYSICS-BASED CONJUNCTION LABELLING
=======================================================================================
 Builds the ground truth for the collision-risk project.

 Instead of inventing a risk label from a formula over the same features the model
 later sees (which is circular - the model just reverse-engineers the formula), this
 stage derives the label from actual orbital mechanics:

   1. Parse real TLEs for Starlink satellites and COSMOS 2251 debris.
   2. Propagate every object with SGP4 over a common time window.
   3. Pre-screen object pairs with an analytic apogee/perigee filter.
      This is the O(n^2) reduction step - we never evaluate all pairs.
   4. For each surviving candidate pair, compute the TRUE minimum miss
      distance over the propagation window.

 Output: conjunction_cache.npz  (node features + candidate pairs + miss distances)

 The miss distance depends on all six orbital elements AND the relative time phasing
 of the two objects, so no two-variable algebraic shortcut reproduces it.
=======================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------- CONFIG ------------------------------------
STARLINK_TLE    = "STARLINK-1008 TLE.txt"       # 10,731 distinct Starlink satellites
DEBRIS_TLE      = "COSMOS 2251 Debris TLE.txt"  # 586 COSMOS 2251 debris objects

STARLINK_SAMPLE = 4000      # satellites sampled from the constellation
PROP_HOURS      = 24.0      # propagation window length
STEP_MINUTES    = 2.0       # time step
REFINE_TOP_N    = 60000     # closest pairs that get the fine refinement pass
REFINE_STEP_SEC = 5.0       # fine time step for refinement
REFINE_HALF_WINDOW_MIN = 2.0  # bracket around the coarse TCA, minutes
SCREEN_PAD_KM   = 60.0      # altitude-shell padding in the analytic pre-screen
MAX_CANDIDATES  = 120000    # cap on candidate pairs (random, seeded, unbiased)
ORBIT_SAMPLES   = 150       # true-anomaly samples per orbit for the MOID pass
RNG_SEED        = 42
EARTH_R         = 6378.137  # km
CACHE           = "conjunction_cache.npz"
# -------------------------------------------------------------------------------


def parse_tle_file(path):
    """Read a 3-line (name / line1 / line2) TLE file."""
    with open(path, "r") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    out = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name, l1, l2))
    return out


def build_satrecs(records, label):
    """Turn TLE text into SGP4 Satrec objects.

    Element parsing is done by the SGP4 library itself, so there is no
    hand-written column slicing that can silently go wrong.
    """
    sats, names, bad = [], [], 0
    for name, l1, l2 in records:
        try:
            s = Satrec.twoline2rv(l1, l2)
            sats.append(s)
            names.append(name)
        except Exception:
            bad += 1
    print("   {:<22} parsed {:>6} objects   ({} rejected)".format(label, len(sats), bad))
    return sats, names


def node_features(sats):
    """Per-object feature table, read straight off the Satrec structures."""
    rows = []
    for s in sats:
        sma = s.a * EARTH_R                      # semi-major axis, km
        ecc = s.ecco
        rows.append({
            "inclination_deg": np.degrees(s.inclo),
            "eccentricity":    ecc,
            "raan_deg":        np.degrees(s.nodeo),
            "argp_deg":        np.degrees(s.argpo),
            "mean_anom_deg":   np.degrees(s.mo),
            "mean_motion_rpd": s.no_kozai * 1440.0 / (2.0 * np.pi),
            "bstar":           s.bstar,
            "sma_km":          sma,
            "perigee_alt_km":  sma * (1.0 - ecc) - EARTH_R,
            "apogee_alt_km":   sma * (1.0 + ecc) - EARTH_R,
        })
    return pd.DataFrame(rows)


def time_grid(sats, hours, step_minutes):
    """Common propagation window, anchored at the latest TLE epoch in the set."""
    epochs = np.array([s.jdsatepoch + s.jdsatepochF for s in sats])
    start = float(epochs.max())
    n = int(round(hours * 60.0 / step_minutes)) + 1
    offs = np.arange(n) * (step_minutes / 1440.0)
    jd = np.full(n, start)
    return jd, offs, start, n


def propagate(sats, jd, fr, tag):
    """Vectorised SGP4 propagation. Returns (n_obj, n_time, 3) TEME positions in km."""
    t0 = time.time()
    arr = SatrecArray(sats)
    err, r, _v = arr.sgp4(jd, fr)
    ok = (err == 0).all(axis=1)
    print("   {}: propagated {} objects x {} epochs in {:.1f}s   ({} dropped on error)"
          .format(tag, len(sats), len(jd), time.time() - t0, int((~ok).sum())))
    return r.astype(np.float32), ok


def orbit_samples(feat, n_samples):
    """Sample each object's orbital PATH in inertial space.

    Returns (n_obj, n_samples, 3) km: the geometric ellipse each object traces,
    sampled uniformly in true anomaly. Time is deliberately absent - this is the
    orbit as a curve, not the object as a moving point.
    """
    a = feat["sma_km"].to_numpy()[:, None]
    e = feat["eccentricity"].to_numpy()[:, None]
    inc = np.radians(feat["inclination_deg"].to_numpy())[:, None]
    raan = np.radians(feat["raan_deg"].to_numpy())[:, None]
    argp = np.radians(feat["argp_deg"].to_numpy())[:, None]

    nu = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)[None, :]
    r = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))

    # perifocal frame
    xp, yp = r * np.cos(nu), r * np.sin(nu)

    # rotate perifocal -> inertial:  Rz(raan) Rx(inc) Rz(argp)
    ca, sa = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc), np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)

    x1 = xp * ca - yp * sa
    y1 = xp * sa + yp * ca
    # Rx(inc)
    y2 = y1 * ci
    z2 = y1 * si
    # Rz(raan)
    X = x1 * cr - y2 * sr
    Y = x1 * sr + y2 * cr
    Z = np.broadcast_to(z2, X.shape)

    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def compute_moid(S, pa, pb, chunk=200):
    """Minimum Orbital Intersection Distance for each candidate pair.

    MOID is the closest the two ORBITS ever come, regardless of where the two
    objects happen to be along them. It is the standard first-stage geometric
    screen in real conjunction assessment: a pair whose MOID is large can never
    collide, whatever the phasing.

    Unlike the time-phased miss distance, MOID is a smooth deterministic
    function of the two objects' shape-and-orientation elements (a, e, i, raan,
    argp), which makes it a genuinely learnable target - while still requiring
    all ten values in a strongly non-linear combination, so no two-variable
    shortcut reproduces it.
    """
    m = len(pa)
    out = np.empty(m, dtype=np.float32)
    t0 = time.time()
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        A = S[pa[s:e]][:, :, None, :]       # (c, N, 1, 3)
        B = S[pb[s:e]][:, None, :, :]       # (c, 1, N, 3)
        d = A - B
        dist = np.sqrt(np.einsum("cijk,cijk->cij", d, d))
        out[s:e] = dist.reshape(e - s, -1).min(axis=1)
        if (s // chunk) % 100 == 0:
            print("      {:>8}/{}  ({:5.1f}s)".format(e, m, time.time() - t0))
    return out


def analytic_screen(feat, idx_a, idx_b, pad_km):
    """Apogee/perigee shell-overlap filter.

    Two objects can only ever conjunct if their radial bands overlap. This is a
    cheap algebraic test that removes the large majority of pairs before any
    propagation is used - the step that makes the problem tractable at scale.
    """
    rp = (feat["perigee_alt_km"].to_numpy() + EARTH_R) - pad_km
    ra = (feat["apogee_alt_km"].to_numpy() + EARTH_R) + pad_km
    A, B = np.meshgrid(idx_a, idx_b, indexing="ij")
    A, B = A.ravel(), B.ravel()
    keep = (rp[A] <= ra[B]) & (rp[B] <= ra[A])
    return A[keep], B[keep]


def refine_min_distance(sats, start_jd, fr, pa, pb, md, at,
                        top_n, step_sec, half_window_min):
    """Second-pass refinement of the closest candidate pairs.

    The coarse grid samples every STEP_MINUTES. At LEO relative velocities
    (~7.5 km/s) two minutes of flight is ~900 km, so a coarse minimum badly
    overestimates the true closest approach. Here the closest `top_n` pairs are
    re-propagated on a fine grid bracketing their coarse time of closest
    approach, which recovers the real minimum.

    Pairs are grouped by their coarse TCA index so each group only needs the
    objects it actually involves - this keeps memory flat instead of
    propagating the whole catalogue at 5-second resolution.
    """
    md = md.copy()
    order = np.argsort(md)[:min(top_n, len(md))]
    tca = at[order]

    half = half_window_min / 1440.0
    dt = step_sec / 86400.0
    offs = np.arange(-half, half + dt * 0.5, dt)
    n_fine = len(offs)
    jd_const = np.full(n_fine, start_jd)

    t0 = time.time()
    groups = np.unique(tca)
    improved = 0
    for gi, idx in enumerate(groups):
        sel = order[tca == idx]
        objs = np.unique(np.concatenate([pa[sel], pb[sel]]))
        remap = {int(o): k for k, o in enumerate(objs)}

        local = SatrecArray([sats[int(o)] for o in objs])
        err, r, _v = local.sgp4(jd_const, fr[idx] + offs)
        r = r.astype(np.float32)
        bad = (err != 0).any(axis=1)

        ia = np.array([remap[int(x)] for x in pa[sel]])
        ib = np.array([remap[int(x)] for x in pb[sel]])
        valid = ~(bad[ia] | bad[ib])

        d = r[ia[valid]] - r[ib[valid]]
        fine = np.sqrt(np.einsum("ctk,ctk->ct", d, d)).min(axis=1)

        tgt = sel[valid]
        better = fine < md[tgt]
        improved += int(better.sum())
        md[tgt] = np.minimum(md[tgt], fine)

        if gi % 120 == 0:
            print("      group {:>4}/{}  ({:5.1f}s)".format(gi, len(groups), time.time() - t0))

    print("      refined {:,} pairs on a {:.0f}s grid; {:,} minima improved"
          .format(len(order), step_sec, improved))
    return md


def min_miss_distance(R, pa, pb, chunk=4000):
    """True minimum separation for each candidate pair over the window."""
    m = len(pa)
    best = np.empty(m, dtype=np.float32)
    at = np.empty(m, dtype=np.int32)
    t0 = time.time()
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d = R[pa[s:e]] - R[pb[s:e]]            # (c, T, 3)
        dist = np.sqrt(np.einsum("ctk,ctk->ct", d, d))
        best[s:e] = dist.min(axis=1)
        at[s:e] = dist.argmin(axis=1)
        if (s // chunk) % 15 == 0:
            print("      {:>8}/{}  ({:5.1f}s)".format(e, m, time.time() - t0))
    return best, at


def main():
    print("=" * 87)
    print(" STAGE A - PHYSICS-BASED CONJUNCTION LABELLING (SGP4)")
    print("=" * 87)
    rng = np.random.default_rng(RNG_SEED)

    # ---- 1. load TLEs -------------------------------------------------------
    print("\n[1] Loading TLE catalogues")
    sl_rec = parse_tle_file(os.path.join(HERE, STARLINK_TLE))
    db_rec = parse_tle_file(os.path.join(HERE, DEBRIS_TLE))

    if STARLINK_SAMPLE and STARLINK_SAMPLE < len(sl_rec):
        pick = rng.choice(len(sl_rec), size=STARLINK_SAMPLE, replace=False)
        sl_rec = [sl_rec[i] for i in sorted(pick)]
        print("   sampled {} Starlink satellites (seed={})".format(STARLINK_SAMPLE, RNG_SEED))

    sl_sats, sl_names = build_satrecs(sl_rec, "Starlink")
    db_sats, db_names = build_satrecs(db_rec, "COSMOS 2251 debris")

    sats = sl_sats + db_sats
    names = sl_names + db_names
    is_debris = np.concatenate([np.zeros(len(sl_sats), bool),
                                np.ones(len(db_sats), bool)])

    # ---- 2. propagate -------------------------------------------------------
    print("\n[2] SGP4 propagation  ({:.0f} h window, {:.0f} min step)"
          .format(PROP_HOURS, STEP_MINUTES))
    jd, fr, start_jd, nt = time_grid(sats, PROP_HOURS, STEP_MINUTES)
    R, ok = propagate(sats, jd, fr, "coarse")

    keep = np.where(ok)[0]
    R = R[keep]
    names = [names[i] for i in keep]
    is_debris = is_debris[keep]
    sats = [sats[i] for i in keep]
    feat = node_features(sats)
    print("   surviving objects: {}  ({} Starlink / {} debris)"
          .format(len(sats), int((~is_debris).sum()), int(is_debris.sum())))

    # ---- 3. analytic candidate screening ------------------------------------
    print("\n[3] Analytic apogee/perigee screening  (pad = {:.0f} km)".format(SCREEN_PAD_KM))
    sl_idx = np.where(~is_debris)[0]
    db_idx = np.where(is_debris)[0]
    total_possible = len(sl_idx) * len(db_idx)
    pa, pb = analytic_screen(feat, sl_idx, db_idx, SCREEN_PAD_KM)
    print("   all cross pairs           : {:,}".format(total_possible))
    print("   survived shell overlap    : {:,} ({:.2f}% retained)"
          .format(len(pa), 100.0 * len(pa) / max(total_possible, 1)))

    if len(pa) > MAX_CANDIDATES:
        sel = rng.choice(len(pa), size=MAX_CANDIDATES, replace=False)
        pa, pb = pa[sel], pb[sel]
        print("   randomly capped to        : {:,} candidate pairs".format(len(pa)))

    if len(pa) == 0:
        raise SystemExit("No candidate pairs survived screening - widen SCREEN_PAD_KM.")

    # ---- 4. true minimum miss distance --------------------------------------
    print("\n[4] Coarse minimum miss distance for {:,} pairs".format(len(pa)))
    md_coarse, at = min_miss_distance(R, pa, pb)

    print("\n[4b] Refining closest approaches  ({:.0f}s grid, +/-{:.0f} min bracket)"
          .format(REFINE_STEP_SEC, REFINE_HALF_WINDOW_MIN))
    md = refine_min_distance(sats, start_jd, fr, pa, pb, md_coarse, at,
                             REFINE_TOP_N, REFINE_STEP_SEC, REFINE_HALF_WINDOW_MIN)

    print("\n   TIME-PHASED miss-distance distribution (km):")
    for q in [0.01, 0.1, 1, 5, 25, 50]:
        print("      p{:<6} = {:10.2f}".format(q, np.percentile(md, q)))
    print("      min    = {:10.2f}".format(md.min()))
    print("      median = {:10.2f}".format(np.median(md)))
    for thr in [1, 5, 10, 25, 50, 100, 200]:
        n = int((md < thr).sum())
        print("      < {:>4} km : {:>8,} pairs  ({:7.3f}%)"
              .format(thr, n, 100.0 * n / len(md)))

    # ---- 4c. MOID (timing-independent orbital geometry) ---------------------
    print("\n[4c] Minimum Orbital Intersection Distance ({} samples/orbit)"
          .format(ORBIT_SAMPLES))
    S = orbit_samples(feat, ORBIT_SAMPLES)
    print("      sampled {} orbital paths".format(len(S)))
    moid = compute_moid(S, pa, pb)

    print("\n   MOID distribution (km):")
    for q in [1, 5, 10, 25, 50, 75]:
        print("      p{:<6} = {:10.2f}".format(q, np.percentile(moid, q)))
    print("      min    = {:10.2f}".format(moid.min()))
    print("      median = {:10.2f}".format(np.median(moid)))
    print("      max    = {:10.2f}".format(moid.max()))
    for thr in [1, 5, 10, 25, 50, 100, 200]:
        n = int((moid < thr).sum())
        print("      < {:>4} km : {:>8,} pairs  ({:7.3f}%)"
              .format(thr, n, 100.0 * n / len(moid)))

    # ---- 5. save ------------------------------------------------------------
    out = os.path.join(HERE, CACHE)
    np.savez_compressed(
        out,
        node_feat=feat.to_numpy(np.float64),
        feat_names=np.array(list(feat.columns)),
        names=np.array(names, dtype=object),
        is_debris=is_debris,
        pair_a=pa.astype(np.int32),
        pair_b=pb.astype(np.int32),
        min_dist_km=md,
        min_dist_coarse_km=md_coarse,
        moid_km=moid,
        tca_index=at,
        step_minutes=np.float64(STEP_MINUTES),
        start_jd=np.float64(start_jd),
        n_epochs=np.int64(nt),
    )
    print("\n[5] Saved -> {}".format(CACHE))
    print("=" * 87)


if __name__ == "__main__":
    main()
