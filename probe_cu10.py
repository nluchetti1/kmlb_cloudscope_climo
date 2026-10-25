#!/usr/bin/env python3
"""
Probe: why does the KMLB hybrid find "cumulus colder than -10 C within 5 nmi of LC-39A" in hours
where the MRMS climatology finds none? (90% of the hybrid's extra violations, Oct 2026 analysis.)

For each case hour: the echo the hybrid reads at its -10 C height nearest LC-39A, and
  - dBZ, RHOHV and ZDR at the actual KMLB gate(s) behind it (weather or not?)
  - the beam geometry there: which sweeps bracket the -10 C height, and whether the value was
    interpolated, taken from the nearer beam, or from the top beam's upper half
  - the -10 C height used (freezing level + 6.5 C/km) and the freezing level itself
  - MRMS's own Reflectivity_-10C at that cell (computed by MRMS at ITS -10 C height)
  - the MRMS 3-D column there, 5.0-8.5 km: does MRMS have echo at that height at all?

Runs in the kmlb_cloudscope_climo repo (uses climo_run.py and kmlb_classify.py).
    python probe_cu10.py                     # the 24 built-in cases
    python probe_cu10.py 2024-07-16T05:00    # or your own hours (UTC)
"""
import datetime as dt
import os
import sys
import tempfile

import numpy as np

import climo_run as CR
import kmlb_classify as K

CASES = """2021-07-25T19:00 2021-08-28T16:00 2022-03-08T05:00 2022-03-15T05:00 2022-06-12T03:00
2022-11-28T01:00 2022-12-21T05:00 2023-12-14T16:00 2024-01-03T15:00 2024-01-08T15:00 2024-01-17T17:00
2024-02-04T14:00 2024-03-22T04:00 2024-03-22T17:00 2024-06-18T03:00 2024-07-16T05:00 2024-09-13T04:00
2024-09-14T17:00 2025-06-02T16:00 2025-06-06T01:00 2025-07-09T15:00 2025-07-30T18:00 2025-08-27T03:00
2025-11-07T07:00""".split()
LC39A = K.LC39A
LEVELS_3D = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def full_sweeps(raw, key):
    """Per elevation (best sweep, as in kmlb_classify): az, rng, Z, ZDR, RHOHV."""
    import pyart
    with tempfile.NamedTemporaryFile(suffix=os.path.basename(key)) as fp:
        fp.write(raw)
        fp.flush()
        radar = pyart.io.read_nexrad_archive(fp.name, include_fields=[
            "reflectivity", "differential_reflectivity", "cross_correlation_ratio"])
    best = {}
    for i in range(radar.nsweeps):
        s, e = radar.get_start_end(i)
        el = round(float(radar.fixed_angle["data"][i]), 1)
        g = lambda f: (np.ma.filled(radar.fields[f]["data"][s:e + 1].astype(np.float32), np.nan)
                       if f in radar.fields else None)
        z = g("reflectivity")
        n = int(np.isfinite(z).sum())
        if el not in best or n > best[el]["n"]:
            best[el] = {"el": el, "az": np.asarray(radar.azimuth["data"][s:e + 1]), "n": n,
                        "rng": np.asarray(radar.range["data"]), "z": z,
                        "zdr": g("differential_reflectivity"), "rho": g("cross_correlation_ratio")}
    return [best[k] for k in sorted(best)], float(radar.altitude["data"][0])


def gate_values(sw, rlat, rlon, alt, lat, lon):
    """Z, ZDR, RHOHV at the gate nearest a point on this sweep, plus the 3x3 gate median."""
    s, az = K.column_geometry(np.array([lat]), np.array([lon]), rlat, rlon)
    h, r, w = K.beam(sw["el"], s, alt)
    ray = int(np.argmin(np.abs(((sw["az"] - az[0, 0]) + 180) % 360 - 180)))
    gi = int(round((r[0, 0] - sw["rng"][0]) / (sw["rng"][1] - sw["rng"][0])))
    if not 0 <= gi < sw["rng"].size:
        return None
    rs = [(ray + d) % sw["az"].size for d in (-1, 0, 1)]
    gs = slice(max(gi - 1, 0), gi + 2)
    out = {"h_km": float(h[0, 0]) / 1000, "halfw_km": float(w[0, 0]) / 1000}
    for f in ("z", "zdr", "rho"):
        a = sw[f]
        out[f] = (float(a[ray, gi]) if a is not None else np.nan)
        out[f + "_med"] = (float(np.nanmedian(a[rs][:, gs])) if a is not None and np.isfinite(a[rs][:, gs]).any()
                           else np.nan)
    return out


def branch(H, W, Z, t):
    """How at_heights produced the value at target height t (m): interp / nearer beam / top-beam edge."""
    K_ = len(H)
    idx = int((H <= t).sum()) - 1
    if idx < 0:
        return "below lowest beam (within half width)" if t >= H[0] - W[0] else "not sampled"
    if idx >= K_ - 1:
        return (f"ABOVE top beam centre by {(t - H[-1]) / 1000:.2f} km (top-beam upper half)"
                if t <= H[-1] + W[-1] else "not sampled")
    z1, z2 = Z[idx], Z[idx + 1]
    f = (t - H[idx]) / max(H[idx + 1] - H[idx], 1)
    if z1 > K.NO_ECHO and z2 > K.NO_ECHO:
        return f"interpolated between beams {idx} and {idx + 1} (f={f:.2f})"
    return f"nearer beam ({'lower' if f < 0.5 else 'upper'}; the other has no echo)"


def case(sess, t):
    say(f"\n=== {t:%Y-%m-%d %H:%MZ}  (local ~{(t.hour - 4) % 24:02d}h)")
    get = lambda prod, tol: CR.nearest(sess, prod, t, tol)
    u10, uz = get(CR.ARCHIVE["r10"], 10), get(CR.FREEZING_DIR, 6 * 60)
    if not u10 or not uz:
        say("   MRMS -10C or freezing level missing")
        return None
    r10, la, lo, _ = CR.fetch(sess, u10, -99.0)
    z0, _, _, _ = CR.fetch(sess, uz, -999.0)
    cols = {}
    for h in LEVELS_3D:
        u = get(f"MergedReflectivityQC_{h:05.2f}", 10)
        cols[h] = CR.fetch(sess, u, -999.0)[0] if u else None
    hit = CR.kmlb_nearest(sess, t)
    if not hit:
        say("   no KMLB volume")
        return None
    raw = sess.get(f"{K.BUCKET}/{hit[1]}", timeout=120).content
    vol = K.read_volume(raw, hit[1])
    kf = K.KmlbFrame(vol, la, lo, None, z0, t.isoformat())
    h10 = kf.iso_m[-10.0]
    K10 = K.at_heights(kf.Z, kf.H, kf.W, h10)
    LOg, LAg = np.meshgrid(lo, la)
    dist = np.hypot((LAg - LC39A[0]) * 60, (LOg - LC39A[1]) * 60 * np.cos(np.radians(LC39A[0])))
    near = dist <= 5.0
    kc = near & (K10 > K.NO_ECHO)
    mc = near & (r10 > -90)
    say(f"   KMLB {os.path.basename(hit[1])}, VCP {vol['vcp']}, top {vol['top_deg']} deg; "
        f"cells within 5 nmi with echo at -10 C: hybrid {int(kc.sum())}, MRMS -10C product {int(mc.sum())}")
    if not kc.any():
        say("   (no hybrid echo at -10 C within 5 nmi this hour - the rule fired on a neighbouring frame/feature)")
        return None
    j, i = np.unravel_index(np.where(kc, dist, np.inf).argmin(), dist.shape)
    t10 = float(h10[j, i])
    say(f"   nearest hybrid -10 C echo: {la[j]:.3f}, {lo[i]:.3f} ({dist[j, i]:.1f} nmi from LC-39A); "
        f"{K10[j, i]:.1f} dBZ at {t10 / 1000:.2f} km; freezing level {z0[j, i] / 1000:.2f} km")
    say(f"   how: {branch(kf.H[:, j, i], kf.W[:, j, i], kf.Z[:, j, i], t10)}")
    say("   beams there (centre km +/- half width : sampled dBZ): " + ", ".join(
        f"{vol['sweeps'][k]['el']}: {kf.H[k, j, i] / 1000:.2f}+/-{kf.W[k, j, i] / 1000:.2f}:"
        f"{'--' if not np.isfinite(kf.Z[k, j, i]) else ('none' if kf.Z[k, j, i] <= K.NO_ECHO else f'{kf.Z[k, j, i]:.0f}')}"
        for k in range(kf.Z.shape[0]) if abs(kf.H[k, j, i] - t10) < 3000))
    say(f"   MRMS Reflectivity_-10C there: {r10[j, i]:.1f}" if r10[j, i] > -90 else "   MRMS Reflectivity_-10C there: no echo")
    say("   MRMS 3-D column there (km: dBZ): " + ", ".join(
        f"{h}: {'--' if cols[h] is None else ('none' if cols[h][j, i] <= -90 else f'{cols[h][j, i]:.0f}')}"
        for h in LEVELS_3D))
    sw, alt = full_sweeps(raw, hit[1])
    gates = []
    for s_ in sw:
        g = gate_values(s_, vol["lat"], vol["lon"], alt, float(la[j]), float(lo[i]))
        if g and abs(g["h_km"] * 1000 - t10) < 1500:
            gates.append((s_["el"], g))
            say(f"   gate {s_['el']:>4} deg at {g['h_km']:.2f} km: Z {g['z']:.1f} (3x3 {g['z_med']:.1f}), "
                f"ZDR {g['zdr']:.2f} ({g['zdr_med']:.2f}), RHOHV {g['rho']:.3f} ({g['rho_med']:.3f})")
    rho_best = np.nanmax([g["rho_med"] for _, g in gates]) if gates else np.nan
    mrms_at = None
    for h in LEVELS_3D:
        if cols[h] is not None and abs(h * 1000 - t10) <= 250:
            mrms_at = cols[h][j, i] > -90
    return {"t": t, "dbz": float(K10[j, i]), "h10": t10, "rho": rho_best,
            "edge": "ABOVE top beam" in branch(kf.H[:, j, i], kf.W[:, j, i], kf.Z[:, j, i], t10),
            "mrms10": bool(r10[j, i] > -90), "mrms3d_at_h10": mrms_at,
            "night": (t.hour - 4) % 24 >= 21 or (t.hour - 4) % 24 <= 4}


def summarize(res):
    if not res:
        say("\nno case had a hybrid -10 C echo within 5 nmi")
        return
    n = len(res)
    f = lambda k: sum(1 for r in res if r[k]) / n * 100
    say(f"\n=== SUMMARY over {n} cases with a hybrid -10 C echo within 5 nmi")
    say(f"   echo dBZ at -10 C: median {np.median([r['dbz'] for r in res]):.1f}, "
        f"{sum(r['dbz'] < 10 for r in res)} of {n} under 10 dBZ")
    rhos = [r["rho"] for r in res if np.isfinite(r["rho"])]
    say(f"   best 3x3 RHOHV of the gates behind it: median {np.median(rhos):.3f}; "
        f"{sum(x >= 0.9 for x in rhos)} of {len(rhos)} >= 0.90 (weather-like), "
        f"{sum(x < 0.8 for x in rhos)} < 0.80 (non-weather-like)")
    say(f"   value from the top beam's upper half: {f('edge'):.0f}%")
    say(f"   MRMS -10C product has echo at that cell: {f('mrms10'):.0f}%")
    m3 = [r["mrms3d_at_h10"] for r in res if r["mrms3d_at_h10"] is not None]
    say(f"   MRMS 3-D has echo at the hybrid's -10 C height (+/-250 m): "
        f"{sum(m3)} of {len(m3)}")
    say(f"   night cases {sum(r['night'] for r in res)} of {n}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "T" in a]
    times = [dt.datetime.fromisoformat(a) for a in (args or CASES)]
    sess = CR._session()
    res = []
    for t in times:
        try:
            r = case(sess, t)
            if r:
                res.append(r)
        except Exception as e:
            say(f"   !! {type(e).__name__}: {str(e)[:200]}")
    summarize(res)
    open("probe_cu10_report.txt", "w").write("\n".join(OUT) + "\n")
    say("wrote probe_cu10_report.txt")
