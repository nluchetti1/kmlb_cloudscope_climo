"""GLM in the climatology: the real climo_run.main() + climo_merge.py over synthetic hours, with
flashes from Cape GLM archive files (as glm_prep.py writes them) served at GLM_URL. Runs in either
climatology repo.    python test_glm.py"""
import datetime as dt, io, json, os, shutil, sys, tempfile
from contextlib import redirect_stdout
import numpy as np
sys.path.insert(0, os.getcwd())
import climo_run as CR, mrms_classify as mc, climo_merge as CM
try:
    import kmlb_classify as K
except ImportError:
    K = None
FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg)
    if not c: FAILS.append(msg)
LA = np.round(29.595 - 0.01 * np.arange(200), 3); LO = np.round(-81.595 + 0.01 * np.arange(200), 3)
LC39A = (28.6084, -80.6043)
E = lambda t: (t - dt.datetime(1970, 1, 1)).total_seconds()

def archive(first_hour, hours, flashes, missing=lambda st: False):
    """An archive month as glm_prep.py writes it: every 20-second file start, and the flashes."""
    ft = [first_hour + dt.timedelta(seconds=20 * k) for k in range(180 * hours)]
    ft = [t for t in ft if not missing(t)]
    buf = io.BytesIO()
    np.savez_compressed(buf, flash_t=np.array([E(f[2]) for f in flashes], np.float64),
                        flash_lat=np.array([f[0] for f in flashes], np.float32), flash_lon=np.array([f[1] for f in flashes], np.float32),
                        file_t=np.array([int(E(t)) for t in ft], np.int64), file_ok=np.ones(len(ft), bool),
                        satellite=np.array(["G19"] * len(ft)))
    return buf.getvalue()
JUL = dt.datetime(2025, 7, 1)
F15 = [(LC39A[0] + 0.05, LC39A[1], JUL.replace(hour=14, minute=30) + dt.timedelta(seconds=20 * k + s)) for k in range(90) for s in (2, 9, 15)]
F00 = [(LC39A[0] + 0.05, LC39A[1], dt.datetime(2025, 6, 30, 23, 50))]          # the night before
SERVE = {"glm_202507.npz": archive(JUL, 24 * 31, F15, missing=lambda st: st.day == 1 and st.hour == 17 and st.minute >= 30 and (st.second // 20) % 2),
         "glm_202506.npz": archive(dt.datetime(2025, 6, 30), 24, F00)}

class Sess:
    def get(self, url, timeout=None):
        class R:
            def __init__(s, content=b"", code=200): s.content, s.status_code = content, code
            def raise_for_status(s):
                if s.status_code >= 400: raise IOError(f"HTTP {s.status_code}")
        if url.startswith(CR.GLM_URL):
            name = url.rsplit("/", 1)[1]
            return R(SERVE[name]) if name in SERVE else R(code=404)
        return R(b"x")
LOg, LAg = np.meshgrid(LO, LA)
ST = np.hypot((LAg - 29.3) * 111.2, (LOg + 81.3) * 97.7) < 10
def nearest(sess, product, t, tol): return f"fake://{product}/{t:%Y%m%d-%H%M%S}"
def fetch(sess, url, fill):
    prod = url.split("/")[2]
    if prod.startswith("Model_0degC"): f = np.full((200, 200), 4800.0, np.float32)
    elif prod.startswith("MergedReflectivityQC_03"): f = np.where(ST, 20.0, -99.0).astype(np.float32)
    elif prod.startswith("NLDN"): f = np.full((200, 200), -1.0, np.float32)
    else:
        key = next(k for k, p in CR.ARCHIVE.items() if p == prod)
        f = np.where(ST, 20.0 if key not in ("hmax", "vii") else (3000.0 if key == "hmax" else 0.0), mc.PRODUCTS[key][1]).astype(np.float32)
    return f, LA.copy(), LO.copy(), None
CR.nearest, CR.fetch = nearest, fetch
CR.anvil_wind = lambda sess, t: None
CR.sat_tops_bt = lambda sess, t, la, lo, z0: None
CR._session = lambda: Sess()
if K is not None:
    K.list_day = lambda sess, d: []
HOURS = [JUL.replace(hour=0)] + [JUL.replace(hour=h) for h in range(10, 22)]
CR.month_hours = lambda ym, days=None: HOURS
W = tempfile.mkdtemp(prefix="glmh_")
sys.argv = ["climo_run.py", "2025-07", "--days", "31", "--procs", "1", "--out", os.path.join(W, "out")]
CR.main()
z = np.load(os.path.join(W, "out", "202507.npz"))
hrs = [int(t[11:13]) for t in z["t"]]
p = list(mc.SITES).index("LC-39A"); bl = mc.RULE_KEYS.index("lightning")
lit = ((z["red"][:, p] >> bl) & 1).astype(bool)
print(f"   hours {hrs}\n   glm ok {z['glm'].astype(int).tolist()}\n   glm_n {z['glm_n'].tolist()}\n   lightning at LC-39A {lit.astype(int).tolist()}")
check(z["glm"].tolist() == [h != 18 for h in hrs], "GLM usable every hour except 18Z (half its files missing)")
check(lit.tolist() == [h in (0, 15) for h in hrs], "lightning at LC-39A at 15Z and at 00Z (a 23:50 flash from the June file) - NLDN empty")
check(z["glm_n"][hrs.index(15)] == 270 and z["glm_n"][hrs.index(0)] == 1, f"flash counts: 15Z {z['glm_n'][hrs.index(15)]}, 00Z {z['glm_n'][hrs.index(0)]}")
check(str(z["classifier"]).endswith("+glm"), f"classifier recorded as {z['classifier']}")
CR._glm_months.clear(); SERVE.pop("glm_202506.npz")
check(CR.glm_flashes(Sess(), JUL) == ([], 0, False), "previous month's archive file missing -> no GLM for that sample (NLDN only)")
buf = io.StringIO()
sys.argv = ["climo_merge.py", "--inp", os.path.join(W, "out"), "--site", os.path.join(W, "site"), "--web", "web", "--fieldmill", "fieldmill/fieldmill_hourly.npz"]
with redirect_stdout(buf):
    CM.main()
line = next((l for l in buf.getvalue().splitlines() if "match the classifier" in l), "")
print("   " + line.strip())
check("(100.000%)" in line, "merge re-scores lightning from features: 100% agreement with the classifier")
meta = json.load(open(os.path.join(W, "site", "climo.json")))
check(meta["coverage"]["202507"].get("glm") == 12, f"coverage: GLM in {meta['coverage']['202507'].get('glm')} of 13 hours")
check(any("GLM total lightning" in n for n in meta["notes"]), "lightning note says NLDN + GLM")
print(f"\n{len(FAILS)} failure(s)"); sys.exit(1 if FAILS else 0)
