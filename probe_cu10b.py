#!/usr/bin/env python3
"""
Probe 2: WHICH ROUTE makes the hybrid's extra "cumulus colder than -10 C within 5 nmi of LC-39A"?

The class has three routes (mrms_classify.evaluate, v19):
    r10   echo of 0 dBZ or more at the -10 C slice
    gap   echo, no echo at ANY isotherm slice, and the column's strongest return (hmax) at or
          above the -10 C height
    sat   a satellite top colder than -10 C within 3 km above the radar echo top
In the hybrid the slices come from KMLB where it samples them, while composite and hmax may stay
MRMS's (or the larger of the two) where KMLB does not cover the whole column - so the gap route
can see "echo, no slice hits, peak above -10 C" built from two radars. This probe reruns the real
classifier on each hour twice - the MRMS climatology's inputs and the hybrid's - and for the
hybrid's cumulus colder than -10 C nearest LC-39A reports the route and every input from both.

Runs in kmlb_cloudscope_climo.   python probe_cu10b.py [UTC hours ...]
"""
import datetime as dt
import sys

import numpy as np

import climo_run as CR
import kmlb_classify as K
import mrms_classify as mc
from probe_cu10 import CASES, LC39A

OUT = []
SLICES = ("r0", "r5", "r10", "r15", "r20")


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def routes(F, zz, diag):
    """The three cu10 routes as masks, rebuilt exactly as evaluate() builds them."""
    e = {k: F[k] >= 0.0 for k in SLICES}
    any_iso = e["r0"] | e["r5"] | e["r10"] | e["r15"] | e["r20"]
    z_of = lambda tc: zz + (-tc) / mc.LAPSE_C_PER_KM * 1000.0
    echo = diag["echo"]
    aloft = echo & ~any_iso & ((F["hmax"] >= z_of(-20.0)) | (F["super"] >= 0.0))
    gap = echo & ~any_iso & ~aloft & (F["hmax"] >= z_of(0.0))
    gap_m10 = gap & (F["hmax"] >= z_of(-10.0))
    not_cu = diag["attached"] | diag["detached"] | diag["elevated"] | diag["layered"]
    return {"r10": e["r10"] & ~not_cu, "gap": gap_m10 & ~not_cu,
            "sat": diag["sat_up"] & ~(e["r10"] | gap_m10) & ~not_cu}


def fields(sess, t):
    F, la, lo = {}, None, None
    for k, prod in CR.ARCHIVE.items():
        u = CR.nearest(sess, prod, t, 10)
        if u:
            F[k], la, lo, _ = CR.fetch(sess, u, mc.PRODUCTS[k][1])
    for k, (_, none, _) in mc.PRODUCTS.items():
        if k not in F:
            F[k] = np.full((la.size, lo.size), none, np.float32)
    return F, la, lo


def case(sess, t):
    say(f"\n=== {t:%Y-%m-%d %H:%MZ}  (local ~{(t.hour - 4) % 24:02d}h)")
    Fm, la, lo = fields(sess, t)
    uz = CR.nearest(sess, CR.FREEZING_DIR, t, 6 * 60)
    zz = CR.fetch(sess, uz, -999.0)[0] if uz else np.full((la.size, lo.size), 4800.0, np.float32)
    us = CR.nearest(sess, CR.STEINER_DIR, t, 10)
    lvl = CR.fetch(sess, us, -999.0)[0] if us else None
    try:
        sat = CR.sat_tops_bt(sess, t, la, lo, zz)
    except Exception:
        sat = None
    hit = CR.kmlb_nearest(sess, t)
    if not hit:
        say("   no KMLB volume")
        return None
    vol = K.read_volume(sess.get(f"{K.BUCKET}/{hit[1]}", timeout=120).content, hit[1])
    kf = K.KmlbFrame(vol, la, lo, None, zz, t.isoformat())
    Fh = {k: v.copy() for k, v in Fm.items()}
    kf.apply_fields(Fh)
    out = {}
    for name, F, L in (("MRMS", Fm, lvl), ("hybrid", Fh, kf.merge_level(lvl, CR.STEINER_KM) if lvl is not None else None)):
        sep = mc.separation_lite(L, CR.STEINER_KM, la, lo) if L is not None else None
        status, red, yel, cls, top, diag = mc.evaluate(F, zz, la, lo, None, None, sep, None, sat)
        out[name] = (red, cls, diag, routes(F, zz, diag), F)
    LOg, LAg = np.meshgrid(lo, la)
    dist = np.hypot((LAg - LC39A[0]) * 60, (LOg - LC39A[1]) * 60 * np.cos(np.radians(LC39A[0])))
    near = dist <= 5.0
    jp, ip = np.unravel_index(dist.argmin(), dist.shape)
    fired = {n: bool(out[n][0]["cumulus_5nm"][jp, ip]) for n in out}
    cnt = {n: {r: int((m & near).sum()) for r, m in out[n][3].items()} for n in out}
    say(f"   KMLB top {vol['top_deg']} deg; cumulus_5nm at LC-39A: MRMS {fired['MRMS']}, hybrid {fired['hybrid']}")
    say(f"   cu10 cells within 5 nmi by route - MRMS {cnt['MRMS']}  hybrid {cnt['hybrid']}")
    rh = out["hybrid"][3]
    cu = (rh["r10"] | rh["gap"] | rh["sat"]) & near
    res = {"t": t, "fired": fired, "route": None, "mixed": False}
    if not cu.any():
        say("   hybrid has no cu10 cell within 5 nmi (verdict differs from the climatology run)")
        return res
    j, i = np.unravel_index(np.where(cu, dist, np.inf).argmin(), dist.shape)
    route = next(r for r in ("r10", "gap", "sat") if rh[r][j, i])
    res["route"] = route
    Fm_, Fh_ = out["MRMS"][4], out["hybrid"][4]
    ksamp = {k: bool(np.isfinite(K.at_heights(kf.Z, kf.H, kf.W, kf.iso_m[tc])[j, i]))
             for k, tc in K.ISO_KEYS.items()}
    say(f"   nearest hybrid cu10: {la[j]:.3f}, {lo[i]:.3f} ({dist[j, i]:.1f} nmi), route {route.upper()}; "
        f"KMLB samples this column to {kf.top[j, i] / 1000:.1f} km")
    fmt = lambda v: "none" if v <= -90 else f"{v:.0f}"
    say("   field     MRMS    hybrid   (hybrid source)")
    for k in SLICES + ("comp", "super"):
        src = ("KMLB" if ksamp.get(k) else "MRMS") if k in SLICES else "max/KMLB"
        say(f"   {k:6s} {fmt(Fm_[k][j, i]):>7s} {fmt(Fh_[k][j, i]):>8s}   {src}")
    say(f"   hmax   {Fm_['hmax'][j, i]:7.0f} {Fh_['hmax'][j, i]:8.0f}   "
        f"{'MRMS (unchanged)' if Fm_['hmax'][j, i] == Fh_['hmax'][j, i] else 'KMLB'}; -10 C height "
        f"{zz[j, i] + 10 / mc.LAPSE_C_PER_KM * 1000:.0f} m")
    say(f"   MRMS class here: {mc.CLASSES[int(out['MRMS'][1][j, i])]['name'] if hasattr(mc, 'CLASSES') else int(out['MRMS'][1][j, i])}; "
        f"hybrid class: {mc.CLASSES[int(out['hybrid'][1][j, i])]['name'] if hasattr(mc, 'CLASSES') else int(out['hybrid'][1][j, i])}")
    if route == "gap":
        mixed = (Fm_["hmax"][j, i] == Fh_["hmax"][j, i]) and any(ksamp[k] for k in SLICES)
        res["mixed"] = mixed
        say(f"   gap from mixed sources (KMLB slices + MRMS hmax): {mixed}")
    return res


def summarize(res):
    hyb_only = [r for r in res if r["fired"]["hybrid"] and not r["fired"]["MRMS"]]
    say(f"\n=== SUMMARY: {len(res)} cases rerun; classifier reproduces 'hybrid only' in {len(hyb_only)}")
    for rt in ("r10", "gap", "sat", None):
        n = sum(1 for r in hyb_only if r["route"] == rt)
        say(f"   route {str(rt):5s}: {n}")
    say(f"   gap cases built from mixed sources: {sum(1 for r in hyb_only if r['mixed'])}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "T" in a]
    sess = CR._session()
    res = []
    for t in [dt.datetime.fromisoformat(a) for a in (args or CASES)]:
        try:
            r = case(sess, t)
            if r:
                res.append(r)
        except Exception as e:
            say(f"   !! {type(e).__name__}: {str(e)[:200]}")
    summarize(res)
    open("probe_cu10b_report.txt", "w").write("\n".join(OUT) + "\n")
    say("wrote probe_cu10b_report.txt")
