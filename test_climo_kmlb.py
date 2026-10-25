"""Offline harness for the KMLB climatology: the real climo_run.main() over one synthetic day with
the hybrid active, then the real climo_merge.py with the actual field-mill file. The merge's own
check - re-scoring from features reproduces every classifier verdict - must read 100%.

    python test_climo_kmlb.py
"""
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import climo_run as CR      # noqa: E402
import kmlb_classify as K   # noqa: E402
import mrms_classify as mc  # noqa: E402
from kmlb_synth import LA, LO, STORM, synth_volume, FULL, TRUNC  # noqa: E402

FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg)
    if not c: FAILS.append(msg)

LOg, LAg = np.meshgrid(LO, LA)
ST = np.hypot((LAg - STORM[0]) * 111.2, (LOg - STORM[1]) * 97.7) < 15
FIELDS = {"comp": 45.0, "r0": 30.0, "r5": 28.0, "r10": 25.0, "r15": 22.0, "r20": 20.0, "super": 15.0,
          "hmax": 3000.0, "vii": 2.0}


def install(kmode):
    def nearest(sess, product, t, tol):
        return f"fake://{product}/{t:%Y%m%d-%H%M%S}"
    def fetch(sess, url, fill):
        prod = url.split("/")[2]
        v = None
        if prod.startswith("Model_0degC"):
            f = np.full((200, 200), 4800.0, np.float32)
        elif prod.startswith("MergedReflectivityQC_03"):
            f = np.where(ST, 25.0, -99.0).astype(np.float32)
        elif prod.startswith("NLDN"):
            f = np.full((200, 200), -1.0, np.float32)
        else:
            key = next(k for k, p in CR.ARCHIVE.items() if p == prod)
            none = mc.PRODUCTS[key][1]
            f = np.where(ST, FIELDS[key], none).astype(np.float32)
        return f, LA.copy(), LO.copy(), v
    CR.nearest = nearest
    CR.fetch = fetch
    CR.anvil_wind = lambda sess, t: None
    CR.sat_tops_bt = lambda sess, t, la, lo, z0: None
    class Sess:
        def get(self, url, timeout=None):
            class R:
                content = b"volume"
                def raise_for_status(self): pass
            return R()
    CR._session = lambda: Sess()
    CR._kmlb_days.clear()
    day = dt.date(2025, 7, 31)
    vols = [(dt.datetime(2025, 7, 31) + dt.timedelta(minutes=5 * k + 1),) for k in range(288)]
    vols = [(t, f"2025/07/31/KMLB/KMLB{t:%Y%m%d_%H%M%S}_V06") for (t,) in vols]
    K.list_day = lambda sess, d: (vols if d == day and kmode != "off" else [])
    # AVSET off 13-16Z (as on the Crew-11 day), truncated otherwise
    K.read_volume = lambda raw, key: (lambda t: synth_volume(FULL if 13 <= t.hour < 16 else TRUNC, start=t))(K.vol_start(key))


def run_month(kmode, outdir):
    install(kmode)
    sys.argv = ["climo_run.py", "2025-07", "--days", "31", "--procs", "1", "--out", outdir]
    CR.month_hours_orig = CR.__dict__.get("month_hours_orig", CR.month_hours)
    CR.month_hours = lambda ym, days=None: [dt.datetime(2025, 7, 31, h) for h in range(10, 22)]
    CR.main()
    return np.load(os.path.join(outdir, "202507.npz"), allow_pickle=False)


W = tempfile.mkdtemp(prefix="kclimo_")
print("== KMLB hybrid: 12 hours on 31 Jul 2025 (AVSET off 13-16Z)")
z = run_month("on", os.path.join(W, "on"))
print(f"   {len(z['t'])} samples; k_found {z['k_found'].astype(int).tolist()}; k_top {z['k_top'].tolist()}")
check(len(z["t"]) == 12 and z["k_found"].all(), "every hour scored, every hour with a KMLB volume")
full = (z["k_lc39a"] >= 150)
# each hour takes the volume whose midpoint is nearest: 14:00Z -> the 13:56 volume (full), 13:00Z -> 12:56 (truncated)
check(full.sum() == 3 and [int(t[11:13]) for t in z["t"][full]] == [14, 15, 16], f"full depth over LC-39A at 14-16Z (volumes starting 13:56-15:56): {z['t'][full].tolist()}")
check(set(z["k_top"][~full].tolist()) == {64} and (z["k_lc39a"][~full] == 68).all(), "other hours: 6.4 deg, LC-39A sampled to 6.8 km")
check(str(z["classifier"]) == "mrms-v19+kmlb-v1", f"classifier recorded as {z['classifier']}")
check((np.abs(z["k_off"]) <= 30).all(), f"volume midpoints within 3 min of the samples: {sorted(set(z['k_off'].tolist()))}")

print("\n== same hours with no KMLB volumes: MRMS only")
zo = run_month("off", os.path.join(W, "off"))
check(len(zo["t"]) == 12 and not zo["k_found"].any(), "every hour still scored, none with KMLB")
diff = int((z["feat"] != zo["feat"]).any(axis=2).sum())
check(diff > 0, f"the hybrid changes the recorded features in {diff} of {z['feat'].shape[0] * z['feat'].shape[1]} pad-samples")
print(f"   status, hybrid vs MRMS only, first sample: {z['status'][0].tolist()} vs {zo['status'][0].tolist()}")

print("\n== merge with the real field-mill file")
# month files as the MRMS climatology writes them: no KMLB columns at all
os.makedirs(os.path.join(W, "mrms"), exist_ok=True)
np.savez_compressed(os.path.join(W, "mrms", "202507.npz"),
                    **{k: (np.array("mrms-v19") if k == "classifier" else zo[k])
                       for k in zo.files if not k.startswith("k_")})
for label, d in (("hybrid", "on"), ("MRMS climatology file", "mrms")):
    site = os.path.join(W, f"site_{d}")
    buf = io.StringIO()
    import climo_merge as CM
    sys.argv = ["climo_merge.py", "--inp", os.path.join(W, d), "--site", site, "--web", "web",
                "--fieldmill", "fieldmill/fieldmill_hourly.npz"]
    with redirect_stdout(buf):
        CM.main()
    out = buf.getvalue()
    line = next((l for l in out.splitlines() if "match the classifier" in l), "")
    print("   " + label + ": " + line.strip())
    check("(100.000%)" in line, f"{label}: re-scoring from features matches the classifier 100%")
    meta = json.load(open(os.path.join(site, "climo.json")))
    if d == "on":
        cov = meta["coverage"]["202507"]
        check(meta["kmlb"] is True and cov["kmlb"] == 12 and cov["kmlb_full"] == 3, f"coverage: {cov}")
        check(any("KMLB + MRMS hybrid" in n for n in meta["notes"]), "KMLB caveats in the notes")
        check(meta["classifier"] == ["mrms-v19+kmlb-v1"], f"classifier {meta['classifier']}")
        shutil.rmtree("/tmp/kclimo_site", ignore_errors=True); shutil.copytree(site, "/tmp/kclimo_site")
    else:
        check(meta["kmlb"] is False and not any("hybrid" in n for n in meta["notes"]) and "kmlb" not in meta["coverage"]["202507"],
              "MRMS climatology files: no KMLB notes or coverage - one merge serves both repos")
        shutil.rmtree("/tmp/kclimo_site_mrms", ignore_errors=True); shutil.copytree(site, "/tmp/kclimo_site_mrms")

print("\n== kmlb_nearest")
install("on")
hit = CR.kmlb_nearest(None, dt.datetime(2025, 7, 31, 14, 0))
check(hit and hit[0] == dt.datetime(2025, 7, 31, 13, 56), f"14:00Z -> volume starting {hit and hit[0]:%H:%M} (midpoint 13:58:30)")
check(CR.kmlb_nearest(None, dt.datetime(2025, 8, 1, 12, 0)) is None, "a day with no listing -> None (MRMS only)")
print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
