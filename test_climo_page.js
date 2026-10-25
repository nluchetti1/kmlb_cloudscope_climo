// Node test for the KMLB climatology page: stubbed DOM, real page JS, the harness's merged site.
//   node test_climo_page.js [site] [mrms_site]
const fs = require("fs"), vm = require("vm"), path = require("path");
const SITE = process.argv[2] || "site", MSITE = process.argv[3] || "site_mrms";
let html = fs.readFileSync(path.join(__dirname, "web/index.html"), "utf8");
html = html.replace('const MRMS_CLIMO_URL = "";', 'const MRMS_CLIMO_URL = "mrms";');
const js = html.split("<script>")[1].split("</script>")[0];
const els = {};
function ctx2d(){ const b = { createImageData: (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h * 4)) }),
  measureText: t => ({ width: t.length * 6 }) };
  return new Proxy(b, { get: (t, k) => (k in t ? t[k] : () => ({})), set: (t, k, v) => (t[k] = v, true) }); }
function mk(id){ const el = { id, innerHTML: "", textContent: "", hidden: false, value: 0, style: {}, width: 1200, height: 520,
  clientWidth: 600, classList: { add(){}, remove(){}, toggle(){} }, setAttribute(){}, addEventListener(){}, querySelectorAll: () => [], querySelector: () => null,
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 600, height: 300 }) };
  const c = ctx2d(); el.getContext = () => c; return el; }
const serve = async url => { const u = url.split("?")[0];
  const p = u.startsWith("mrms/") ? path.join(MSITE, u.slice(5)) : path.join(SITE, u);
  if (!fs.existsSync(p)) return { ok: false, status: 404, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0) };
  const b = fs.readFileSync(p); return { ok: true, json: async () => JSON.parse(b),
    arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.length) }; };
const ctx = { document: { getElementById: id => els[id] || (els[id] = mk(id)), querySelectorAll: () => [], createElement: () => mk("x") },
  console, Math, Date, JSON, Promise, Uint8Array, Uint16Array, Float32Array, Infinity, isFinite, parseInt, Number, String,
  Object, Array, Set, Map, Blob, Response, DecompressionStream, navigator: {}, setTimeout, fetch: serve,
  Intl, addEventListener(){}, matchMedia: () => ({ matches: false }) };
ctx.window = ctx; vm.createContext(ctx); vm.runInContext(js, ctx);
const strip = h => String(h).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
setTimeout(() => {
  let bad = 0; const ok = (c, m) => { console.log((c ? "  ok    " : "  FAIL  ") + m); if (!c) bad++; };
  const M = vm.runInContext("M", ctx);
  ok(M && M.kmlb === true, "climo.json loaded, kmlb: true");
  ok(/KMLB Level II and MRMS in single-source columns/.test(els.notes.innerHTML) && /Page kmlb-climo-v2/.test(els.notes.innerHTML), "notes: hybrid description and page version");
  ok(/KMLB \+ MRMS hybrid/.test(els.notes.innerHTML), "caveats include the KMLB notes");
  ok(/KMLB volume 100% · full depth over the pads 25%/.test(strip(els.cov.innerHTML)), "coverage: " + (strip(els.cov.innerHTML).match(/KMLB volume[^<]*/) || [""])[0]);
  ok(vm.runInContext("MREF", ctx) !== null, "MRMS climatology loaded for comparison");
  ok(/MRMS climatology \d+(\.\d)?%/.test(strip(els["g-viol"].innerHTML)), "headline: " + strip(els["g-viol"].innerHTML));
  ok(/MRMS \d+(\.\d)?%/.test(strip(els["g-rules"].innerHTML)), "rule list shows the MRMS share beside each rule");
  ok(/mrms-v19\+kmlb-v2/.test(els.samples.textContent), "header: " + els.samples.textContent);
  console.log(`\n${bad} failure(s)`); process.exit(bad ? 1 : 0);
}, 500);
