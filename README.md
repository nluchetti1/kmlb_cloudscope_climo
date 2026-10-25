# CloudScope KMLB climatology

Every hour from 14 Oct 2020, scored with the KMLB + MRMS hybrid (KMLB Level II wherever its beams
sample a height, MRMS above) and the SAME pinned classifier, isotherms, NLDN lightning and field
mills as the MRMS climatology - so the two records differ only in the radar.

## Layout
```
kmlb_cloudscope_climo/
├── climo_run.py            scores one month (MRMS climatology's script + the KMLB hybrid)
├── climo_merge.py          merges months, publishes the site (same script works for MRMS-only months)
├── mrms_classify.py        pinned classifier - byte-identical to the MRMS climatology's (v19)
├── kmlb_classify.py        pinned KMLB module (hybrid merge + clutter screens) from the live KMLB repo
├── requirements.txt        adds arm_pyart
├── _checks.py
├── fieldmill/
│   └── fieldmill_hourly.npz
├── web/
│   └── index.html          kmlb-climo-v1; set MRMS_CLIMO_URL near the top for side-by-side figures
├── .github/workflows/
│   └── climo.yml
└── tests (optional): test_climo_kmlb.py, test_climo_page.js, kmlb_synth.py
```

## First run
1. Commit, then Settings -> Pages -> deploy from `gh-pages` (created by the first merge).
2. Actions -> "CloudScope KMLB climatology" -> Run with start `2025-07`, end `2025-07`: one month,
   about an hour. Check the score log line ("KMLB in N (M full depth over LC-39A)") and the merge's
   "match the classifier (100.000%)".
3. Then start `2020-10`, end `2025-12` (the field-mill record ends 2025-12-31; later months would be
   scored but excluded). The plan skips months already published.
