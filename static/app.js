// Binzy - Smart Waste Segregation Assistant. Vanilla JS single-page app in the Ghumzy (Yatra Kahani) design language.
"use strict";

// ------------------------------------------------------------------ icons (24px stroke glyphs, Ghumzy style)
const P = (d, w = 1.8) => (size = 22, extra = "") =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}><path d="${d}"/></svg>`;
const I = {
  scan: P("M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M7 12h10"),
  camera: P("M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1zM12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"),
  upload: P("M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4M16 8l-4-4-4 4M12 4v12"),
  chart: P("M4 20V10M10 20V4M16 20v-7M22 20H2"),
  sun: P("M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"),
  info: P("M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 16v-5M12 8h.01"),
  sliders: P("M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"),
  x: P("M18 6L6 18M6 6l12 12"),
  check: P("M20 6L9 17l-5-5", 2.2),
  alert: P("M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"),
  help: P("M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"),
  live: P("M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"),
  stop: P("M6 6h12v12H6z"),
  bulb: P("M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"),
  leaf: P("M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10zM2 21c0-3 1.9-5.4 5.1-6"),
  shield: P("M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM9 12l2 2 4-4"),
  globe: P("M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"),
  moon: P("M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"),
  arrow: P("M5 12h14M13 6l6 6-6 6"),
  refresh: P("M3 12a9 9 0 0 1 15.5-6.2L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16M3 21v-5h5"),
  // bins
  paper: P("M6 2h9l5 5v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zM14 2v6h6M8 13h8M8 17h6"),
  plastic: P("M10 2h4v3l2 3v12a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2V8l2-3zM8 12h8"),
  metal: P("M6 5c0-1.7 2.7-3 6-3s6 1.3 6 3v14c0 1.7-2.7 3-6 3s-6-1.3-6-3zM6 5c0 1.7 2.7 3 6 3s6-1.3 6-3M6 12c0 1.7 2.7 3 6 3s6-1.3 6-3"),
  food: P("M12 7c-1.5-2.5-6-2.5-7 1.5-1 3.8 1.5 9.5 4 11.5 1.2 1 2 .5 3 0 1 .5 1.8 1 3 0 2.5-2 5-7.7 4-11.5-1-4-5.5-4-7-1.5zM12 7c0-2 1-4 3-5"),
  trash: P("M3 6h18M8 6V4h8v2M6 6l1 15h10l1-15M10 11v6M14 11v6"),
};
const BIN_ICON = { "Paper & Cardboard": "paper", "Plastic": "plastic", "Metal": "metal", "Food Scraps / Organic": "food",
  "Non-Recyclable / Special Waste": "trash" };
const SHORT = { "Paper & Cardboard": "Paper & Card", "Plastic": "Plastic", "Metal": "Metal", "Food Scraps / Organic": "Food / Organic",
  "Non-Recyclable / Special Waste": "Non-recyclable" };

const logoMark = (s = 34) => `<img src="/static/icon.svg" width="${s}" height="${s}" alt="">`;
const wordmark = (cls = "") => `<span class="wordmark ${cls}"><span class="a">Bin</span><span class="b">zy</span></span>`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const $ = (sel, root = document) => root.querySelector(sel);

// ------------------------------------------------------------------ state
const S = {
  meta: null,
  current: null,        // { blob, url } of the last image scanned
  result: null,
  busy: false,
  live: { stream: null, timer: null, on: false },
  history: [],
  overrides: {},
  stress: { gamma: 1, glare: 0, occlusion: 0, blur: 0 },
};
try { S.history = JSON.parse(sessionStorage.getItem("binzy-history") || "[]"); } catch { S.history = []; }
try { const t = localStorage.getItem("binzy-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* storage blocked */ }

const ROUTES = [
  { id: "scan", label: "Scan", icon: "scan" },
  { id: "insights", label: "Insights", icon: "chart" },
  { id: "stress", label: "Stress test", icon: "sun" },
  { id: "about", label: "How it works", icon: "info" },
];

// ------------------------------------------------------------------ shell
function shell() {
  $("#brand-top").innerHTML = `${logoMark(34)}<span>${wordmark()}<small>Smart waste segregation</small></span>`;
  $("#btn-settings").innerHTML = I.sliders(20);
  $("#btn-settings").onclick = openSettings;
}
function renderNav(route) {
  $("#nav-top").innerHTML = ROUTES.map((r) =>
    `<a href="#/${r.id}" ${r.id === route ? 'aria-current="page"' : ""}>${r.label}</a>`).join("") +
    `<button class="btn btn-primary btn-sm" style="margin-left:.5rem" id="nav-scan">${I.camera(18)} Scan now</button>`;
  const tab = (r) => `<a class="tab" href="#/${r.id}" ${r.id === route ? 'aria-current="page"' : ""}><span class="pill">${I[r.icon](22)}</span>${r.label.split(" ")[0]}</a>`;
  $("#tabbar").innerHTML = tab(ROUTES[0]) + tab(ROUTES[1]) +
    `<button class="fab" id="fab" aria-label="Scan with camera">${I.camera(26)}</button>` + tab(ROUTES[2]) + tab(ROUTES[3]);
  $("#fab").onclick = () => quickScan();
  $("#nav-scan").onclick = () => quickScan();
}
function quickScan() {
  if (location.hash !== "#/scan" && location.hash !== "") { location.hash = "#/scan"; setTimeout(startCamera, 120); }
  else startCamera();
}

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg; document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

function sheet(title, html) {
  const root = $("#sheet-root");
  root.innerHTML = `<div class="scrim" data-close></div>
    <div class="sheet" role="dialog" aria-modal="true" aria-label="${esc(title)}" tabindex="-1">
      <div class="grab"></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem">
        <h2 style="font-size:1.15rem">${esc(title)}</h2>
        <button class="round-btn" data-close aria-label="Close">${I.x(20)}</button>
      </div>${html}</div>`;
  const close = () => { root.innerHTML = ""; document.removeEventListener("keydown", onKey); };
  const onKey = (e) => e.key === "Escape" && close();
  document.addEventListener("keydown", onKey);
  root.querySelectorAll("[data-close]").forEach((el) => (el.onclick = close));
  $(".sheet", root).focus();
  return { root: $(".sheet", root), close };
}

// ------------------------------------------------------------------ API
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}
async function classify(blob, extra = {}) {
  const fd = new FormData();
  fd.append("image", blob, "image.jpg");
  for (const [k, v] of Object.entries({ ...S.overrides, ...extra })) fd.append(k, v);
  const r = await fetch("/api/predict", { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Error ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------ SCAN view
function greet() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
}

function viewScan() {
  const m = S.meta;
  const hr = m.hybrid_report;
  const ext = hr && (hr.sets.external2 || hr.sets.trashnet);
  return `
  <header class="hero reveal">
    <div class="phone-top">
      <a href="#/scan" class="brand" style="color:#fff">${logoMark(30)}<span class="wordmark" style="font-size:1.15rem">Bin<span style="color:#ffc27a">zy</span></span></a>
      <button class="glass-btn" id="hero-settings" aria-label="Settings">${I.sliders(20)}</button>
    </div>
    <p class="greet">${greet()} 👋</p>
    <h1>Which bin does it go in?</h1>
    <button class="search" id="hero-search">
      ${I.scan(20, 'style="color:var(--brand-700);flex-shrink:0"')}
      <span class="t">Snap or upload any waste item</span>
      <span class="go">${I.camera(18)}</span>
    </button>
    <svg class="hills" viewBox="0 0 320 120" aria-hidden="true"><path d="M0 120V80l50-30 40 20 60-50 50 40 40-15 80 55v20z" fill="#fff"/></svg>
  </header>

  <div class="stagger">
    <section class="sec px">
      <div class="grid2 grid2-wide">
        <div class="card scanner">
          <div class="drop" id="drop">${dropContent()}</div>
          <div class="scan-actions">
            <button class="btn btn-primary" id="btn-camera">${I.camera(20)} <span id="cam-label">${S.live.stream ? "Capture" : "Open camera"}</span></button>
            <button class="btn btn-ghost" id="btn-upload">${I.upload(20)} Upload photo</button>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:.75rem;gap:.5rem;flex-wrap:wrap">
            <button class="chip ${S.live.on ? "chip-on" : ""}" id="btn-live" aria-pressed="${S.live.on}">${I.live(16)} Live auto-scan</button>
            <span class="muted" style="font-size:.78rem">Photos stay on this device's server · nothing is saved</span>
          </div>
        </div>
        <div id="result">${S.result ? resultCard(S.result) : resultPlaceholder()}</div>
      </div>
    </section>

    <section class="sec px">
      <a href="#/about" class="banner">
        <div style="position:relative;z-index:1;max-width:62%">
          <p class="kicker">${m.hybrid_report ? m.hybrid_report.vocabulary.waste_items : "250"}+ items understood</p>
          <p class="title">Recognises everyday waste it was never trained on - wrappers, cake, kulhads, tetra packs</p>
          <span class="cta">How it works ${I.arrow(14)}</span>
        </div>
        <svg class="art" viewBox="0 0 200 140" aria-hidden="true">
          <path d="M0 140V100l40-22 34 12 46-30 40 18 40-10v72z" fill="#d6e6b0"/>
          <rect x="70" y="40" width="70" height="86" rx="10" fill="#4e7a1a"/><rect x="62" y="30" width="86" height="14" rx="7" fill="#2f4a10"/>
          <path d="M92 70c-6 6-8 13-6 20 5-1 10-5 12-10M92 70c3 6 3 13 1 22" stroke="#e6f0c8" stroke-width="4" fill="none" stroke-linecap="round"/>
          <circle cx="160" cy="28" r="12" fill="#e2791b"/>
        </svg>
      </a>
    </section>

    ${m.demos.length ? `
    <section class="sec">
      <div class="sec-head px"><h2 class="section-title">Try a sample</h2><span class="see-all">${m.demos.length} photos</span></div>
      <div class="row px">${m.demos.map((d) => `
        <button class="card demo-card" data-demo="${esc(d.file)}"><img loading="lazy" src="/demo/${encodeURIComponent(d.file)}" alt=""><div>${esc(d.name)}</div></button>`).join("")}
      </div>
    </section>` : ""}

    <section class="sec">
      <div class="sec-head px"><h2 class="section-title">The 5 cafeteria bins</h2><span class="see-all">Tap for details</span></div>
      <div class="row px">${m.categories.map((c) => `
        <button class="bin-tile" data-bin="${esc(c.name)}"><span class="ic" style="background:${c.color}">${I[BIN_ICON[c.name]](28)}</span><b>${esc(SHORT[c.name])}</b></button>`).join("")}
      </div>
    </section>

    ${S.history.length ? `
    <section class="sec px">
      <div class="sec-head"><h2 class="section-title">This session</h2><button class="see-all" id="clear-hist">Clear</button></div>
      <div class="card" style="padding:1rem">${historyBars()}</div>
    </section>` : ""}
  </div>`;
}

function dropContent() {
  if (S.live.stream) return `<video id="video" autoplay playsinline muted></video>
    <span class="live-badge"><i></i>${S.live.on ? "Live scanning" : "Camera on"}</span>${S.live.on ? '<div class="scanline"></div>' : ""}`;
  if (S.current) return `<img src="${S.current.url}" alt="Item being classified">${S.busy ? '<div class="scanline"></div>' : ""}`;
  return `<div class="empty"><div class="big-ic">${I.camera(30)}</div>
    <b style="color:var(--ink)">Drop a photo here</b><br><span style="font-size:.88rem">One item, filling the frame, works best</span></div>`;
}

function resultPlaceholder() {
  return `<div class="card result" style="height:100%;display:flex;flex-direction:column;justify-content:center;gap:.9rem">
    <p class="label">Your result</p>
    <h2 style="font-size:1.3rem">Show me a waste item and I'll tell you the right bin, how to dispose of it - and when I'm not sure.</h2>
    <div class="chips">${S.meta.categories.map((c) => `<span class="chip"><span style="width:.6rem;height:.6rem;border-radius:99px;background:${c.color}"></span>${esc(SHORT[c.name])}</span>`).join("")}</div>
  </div>`;
}

function steps(text) {
  return `<ol class="steps">${text.split("\n").map((l) => `<li>${esc(l.replace(/^\d+\.\s*/, ""))}</li>`).join("")}</ol>`;
}

function catBars(probs) {
  return `<div class="bars">${S.meta.categories.map((c) => `
    <div class="bar"><span>${esc(SHORT[c.name])}</span><span class="track"><span class="fill" style="width:${(probs[c.name] * 100).toFixed(1)}%;background:${c.color}"></span></span><span class="v">${pct(probs[c.name], 0)}</span></div>`).join("")}</div>`;
}

function resultCard(r) {
  const color = r.color;
  const metrics = `<div class="metrics">
      <div class="metric"><span>Confidence</span><b>${pct(r.confidence, 0)}</b></div>
      <div class="metric"><span>Uncertainty</span><b>${r.entropy.toFixed(2)}</b></div>
      <div class="metric"><span>Time</span><b>${Math.round(r.latency_ms)} ms</b></div></div>`;
  const details = `<details class="more"><summary>Why this answer? ›</summary>
      <p class="label" style="margin-top:.8rem">Probability per bin (fused)</p>${catBars(r.category_probs)}
      <p class="label" style="margin-top:1rem">What it looks like</p>
      <div class="chips" style="margin-top:.4rem">${r.top_items.map((t) => `<span class="chip">${esc(t.item)} · ${pct(t.p, 0)}</span>`).join("")}</div>
      <p class="label" style="margin-top:1rem">Three experts voted</p>
      ${expertTable(r)}
      ${metrics}
    </details>`;

  if (r.status === "out_of_scope") {
    return `<div class="card result reveal"><div class="warn oos">
      <h2>${I.help(22)} This doesn't look like waste</h2>
      <p style="margin-top:.4rem">It looks more like <b>${esc(r.non_waste_guess.replace(/^an? /, ""))}</b>. Point the camera at a single waste item.</p></div>
      ${manualPicker()}${details}</div>`;
  }
  if (r.status === "uncertain") {
    return `<div class="card result reveal"><div class="warn">
      <h2>${I.alert(22)} Not sure - please check</h2>
      <p style="margin-top:.4rem">Best guess: <b>${esc(r.item.replace(/^an? /, ""))}</b> → ${esc(r.category)} (${pct(r.confidence, 0)}), but that's not confident enough to sort by.</p>
      <ol class="steps" style="margin-top:.5rem">
        <li><b>Check the packaging label</b> for a recycling symbol (PET 1, HDPE 2, PAP 20–22, ALU 41).</li>
        <li><b>Retake the photo</b> - one item, centred, hand out of the way, no harsh glare.</li>
        <li><b>Ask cafeteria staff.</b> If unverified, keep separate or consult staff to avoid cross-stream contamination or safety hazards.</li></ol></div>
      ${manualPicker()}${details}</div>`;
  }
  return `<div class="card result reveal">
    <div class="result-head">
      <span class="bin-ic" style="background:${color}">${I[BIN_ICON[r.category]](30)}</span>
      <div style="min-width:0;flex:1"><p class="label">Looks like ${esc(r.item.replace(/^an? /, ""))}</p><h2>${esc(r.category)}</h2></div>
      <span class="ring" style="--p:${(r.confidence * 100).toFixed(0)};--c:${color}"><span>${pct(r.confidence, 0)}</span></span>
    </div>
    <div class="put" style="background:color-mix(in oklab, ${color} 13%, var(--card))">
      <span><span class="label">Put it in</span><br><b style="color:${color}">${esc(r.target)}</b></span>
      ${I.check(26, `style="color:${color}"`)}
    </div>
    ${r.item_tip ? `<div class="tip"><span class="ti">${I.bulb(20)}</span><span>${esc(r.item_tip)}</span></div>` : ""}
    <p class="label" style="margin-top:1rem">How to dispose of it</p>${steps(r.instruction)}
    ${details}
  </div>`;
}

function expertTable(r) {
  const names = { zero_shot: "SigLIP zero-shot (open vocabulary)", linear_probe: "SigLIP + trained head", convnext: "ConvNeXt (fine-tuned)" };
  return `<div class="table-wrap"><table class="table" style="margin-top:.3rem"><thead><tr><th>Expert</th><th>Top bin</th><th>p</th></tr></thead><tbody>
    ${Object.entries(r.experts).map(([k, v]) => {
      const [best, p] = Object.entries(v).sort((a, b) => b[1] - a[1])[0];
      return `<tr><td>${names[k]}</td><td>${esc(SHORT[best])}</td><td>${pct(p, 0)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
}

function manualPicker() {
  return `<p class="label" style="margin-top:1rem">Checked the label? Tell me what it is</p>
    <div class="chips" style="margin-top:.5rem">${S.meta.categories.map((c) =>
      `<button class="chip" data-manual="${esc(c.name)}"><span style="width:.6rem;height:.6rem;border-radius:99px;background:${c.color}"></span>${esc(SHORT[c.name])}</button>`).join("")}</div>
    <div id="manual-out"></div>`;
}

function historyBars() {
  const counts = {};
  S.history.forEach((h) => (counts[h] = (counts[h] || 0) + 1));
  const max = Math.max(...Object.values(counts));
  const color = (k) => (S.meta.categories.find((c) => c.name === k) || {}).color || "var(--amber)";
  return `<div class="bars">${Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `
    <div class="bar"><span>${esc(SHORT[k] || k)}</span><span class="track"><span class="fill" style="width:${(v / max) * 100}%;background:${color(k)}"></span></span><span class="v">${v}</span></div>`).join("")}</div>
    <p class="muted" style="font-size:.78rem;margin-top:.6rem">${S.history.length} scans · counts only, no images kept</p>`;
}

function bindScan() {
  $("#hero-search").onclick = () => startCamera();
  $("#hero-settings").onclick = openSettings;
  $("#btn-upload").onclick = () => $("#file-input").click();
  $("#btn-camera").onclick = () => (S.live.stream ? captureFrame() : startCamera());
  $("#btn-live").onclick = toggleLive;
  const drop = $("#drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("drag"); };
  drop.ondragleave = () => drop.classList.remove("drag");
  drop.ondrop = (e) => {
    e.preventDefault(); drop.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) handleBlob(f);
  };
  document.querySelectorAll("[data-demo]").forEach((b) => (b.onclick = async () => {
    stopCamera();
    const blob = await (await fetch(`/demo/${encodeURIComponent(b.dataset.demo)}`)).blob();
    handleBlob(blob);
    $("#drop").scrollIntoView({ behavior: "smooth", block: "center" });
  }));
  document.querySelectorAll("[data-bin]").forEach((b) => (b.onclick = () => binSheet(b.dataset.bin)));
  const ch = $("#clear-hist");
  if (ch) ch.onclick = () => { S.history = []; saveHistory(); render(); };
  bindResult();
  if (S.live.stream) attachVideo();
}

function bindResult() {
  document.querySelectorAll("[data-manual]").forEach((b) => (b.onclick = () => {
    const c = S.meta.categories.find((x) => x.name === b.dataset.manual);
    document.querySelectorAll("[data-manual]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("#manual-out").innerHTML = `<div class="put reveal" style="background:color-mix(in oklab, ${c.color} 13%, var(--card))">
      <span><span class="label">Thanks - put it in</span><br><b style="color:${c.color}">${esc(c.bin)}</b></span>${I.check(26, `style="color:${c.color}"`)}</div>`;
    S.history.push(c.name); saveHistory();
  }));
}

function saveHistory() { try { sessionStorage.setItem("binzy-history", JSON.stringify(S.history.slice(-200))); } catch { /* ignore */ } }

function setResultHTML() {
  const el = $("#result");
  if (el) { el.innerHTML = S.result ? resultCard(S.result) : resultPlaceholder(); bindResult(); }
}
function setDrop() {
  const d = $("#drop");
  if (d && !S.live.stream) d.innerHTML = dropContent();
}

async function handleBlob(blob, { fromLive = false } = {}) {
  if (!blob.type.startsWith("image/") && blob.type !== "") { toast("Please choose an image file"); return; }
  if (!fromLive) {
    if (S.current) URL.revokeObjectURL(S.current.url);
    S.current = { blob, url: URL.createObjectURL(blob) };
  } else {
    S.current = { blob, url: S.current && S.current.fromLive ? S.current.url : URL.createObjectURL(blob), fromLive: true };
  }
  S.busy = true; setDrop();
  try {
    const r = await classify(blob);
    S.result = r;
    if (!fromLive || r.status === "ok") S.history.push(r.status === "ok" ? r.category : "Asked for help"), saveHistory();
    setResultHTML();
    if (!fromLive && window.innerWidth < 900) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast(e.message || "Could not classify this image");
  } finally {
    S.busy = false; setDrop();
  }
}

// camera ------------------------------------------------------------------
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("#file-capture").click(); // phones over plain http: native camera capture
    return;
  }
  try {
    S.live.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    if (location.hash && location.hash !== "#/scan") return;
    const d = $("#drop");
    if (d) { d.innerHTML = dropContent(); attachVideo(); }
    const l = $("#cam-label"); if (l) l.textContent = "Capture";
  } catch (e) {
    toast("Camera not available - opening file picker");
    $("#file-capture").click();
  }
}
function attachVideo() { const v = $("#video"); if (v && S.live.stream) v.srcObject = S.live.stream; }
function stopCamera() {
  if (S.live.timer) clearInterval(S.live.timer);
  S.live.on = false; S.live.timer = null;
  if (S.live.stream) S.live.stream.getTracks().forEach((t) => t.stop());
  S.live.stream = null;
  const b = $("#btn-live"); if (b) { b.classList.remove("chip-on"); b.setAttribute("aria-pressed", "false"); }
  const l = $("#cam-label"); if (l) l.textContent = "Open camera";
}
function grabFrame() {
  const v = $("#video");
  if (!v || !v.videoWidth) return Promise.resolve(null);
  const c = document.createElement("canvas");
  const s = Math.min(1, 800 / Math.max(v.videoWidth, v.videoHeight));
  c.width = v.videoWidth * s; c.height = v.videoHeight * s;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.88));
}
async function captureFrame() {
  const blob = await grabFrame();
  if (!blob) return;
  stopCamera();
  handleBlob(blob);
}
async function toggleLive() {
  if (S.live.on) {
    clearInterval(S.live.timer); S.live.on = false; S.live.timer = null;
    render(); return;
  }
  if (!S.live.stream) await startCamera();
  if (!S.live.stream) return;
  S.live.on = true;
  render();
  S.live.timer = setInterval(async () => {
    if (S.busy || !S.live.on) return;
    const blob = await grabFrame();
    if (!blob) return;
    S.busy = true;
    try { S.result = await classify(blob); S.current = { blob, url: S.current ? S.current.url : "" }; setResultHTML(); }
    catch { /* skip frame */ } finally { S.busy = false; }
  }, 900);
}

function binSheet(name) {
  const c = S.meta.categories.find((x) => x.name === name);
  sheet(c.name, `
    <div class="put" style="background:color-mix(in oklab, ${c.color} 13%, var(--card));margin-top:0">
      <span style="display:flex;gap:.75rem;align-items:center"><span class="bin-ic" style="width:3rem;height:3rem;border-radius:1rem;display:grid;place-items:center;color:#fff;background:${c.color}">${I[BIN_ICON[c.name]](24)}</span>
      <span><span class="label">Bin</span><br><b style="color:${c.color}">${esc(c.bin)}</b></span></span></div>
    <p class="label" style="margin-top:1rem">Trained classes in this bin</p>
    <div class="chips" style="margin-top:.5rem">${c.classes.map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>
    <p class="muted" style="margin-top:1rem;font-size:.88rem">Plus hundreds of described everyday items recognised by the open-vocabulary model.</p>`);
}

// ------------------------------------------------------------------ INSIGHTS view
function viewInsights() {
  const m = S.meta || {}, cr = m.convnext_report, hr = m.hybrid_report, sp = m.split;
  if (!cr || !sp) {
    const missing = [];
    if (!cr) missing.push("artifacts/evaluation_report.json");
    if (!sp) missing.push("artifacts/split_summary.json");
    if (!hr) missing.push("artifacts/hybrid_report.json");
    return `<section class="sec px"><div class="card result" style="padding:1.5rem">
      <h2 style="font-size:1.3rem">${I.alert(24)} Evaluation reports not loaded</h2>
      <p class="muted" style="margin-top:.6rem">Evaluation reports have not been transferred or generated yet. The app requires:</p>
      <ul style="margin:.6rem 0 .8rem 1.2rem;line-height:1.6">
        ${missing.map((x) => `<li><code>${esc(x)}</code></li>`).join("")}
      </ul>
      <p style="font-size:.9rem">Please copy these reports into the <code>artifacts/</code> directory or run <code>python preflight.py</code> to inspect requirements.</p>
    </div></section>`;
  }
  const kt = hr && hr.sets && hr.sets.kaggle_test ? hr.sets.kaggle_test.hybrid : null;
  const ext = hr && hr.sets ? (hr.sets.external2 || hr.sets.trashnet) : null;
  const extName = hr && hr.sets && hr.sets.external2 ? "unseen external dataset" : "TrashNet (external)";
  const DS = { kaggle_test: "Kaggle test split", trashnet: "TrashNet (external)", external2: "External set #2 (held out)" };
  const oodRefused = hr && hr.ood && hr.ood.imagenette ? pct(hr.ood.imagenette.not_answered, 0)
    : (cr && cr.ood && cr.ood.imagenette ? pct(cr.ood.imagenette.flag_rate, 0) : "–");
  return `
  <section class="sec px reveal">
    <p class="label">Evaluation</p><h1 style="font-size:1.75rem;margin-top:.2rem">How well does it work?</h1>
    <p class="muted" style="margin-top:.4rem;max-width:46rem">Every number on this page is computed by our scripts on images the models never trained on. Bin accuracy = the item went to the right one of the 5 bins.</p>
  </section>
  <section class="sec px stagger"><div class="kpis">
    <div class="card kpi"><b>${pct(kt ? kt.accuracy : (cr.category ? cr.category.accuracy : 0))}</b><span>Bin accuracy · ${cr.test_size ? cr.test_size.toLocaleString() : "–"} test images</span></div>
    <div class="card kpi"><b>${ext && ext.hybrid ? pct(ext.hybrid.accuracy) : "–"}</b><span>Bin accuracy · ${extName}</span></div>
    <div class="card kpi"><b>${oodRefused}</b><span>Non-waste photos refused</span></div>
    <div class="card kpi"><b>${sp.total_images ? sp.total_images.toLocaleString() : "–"}</b><span>Images · 12 classes · 70/15/15</span></div>
  </div></section>

  ${hr ? `<section class="sec px">
    <div class="sec-head"><h2 class="section-title">One model vs. the hybrid</h2></div>
    <div class="card" style="padding:1.1rem">
      <p class="muted" style="font-size:.9rem">ConvNeXt alone is excellent on photos like its training data but drops on other photographers' images. The open-vocabulary SigLIP experts close that gap.</p>
      <div class="chart" style="margin-top:.6rem">${groupedBars(Object.keys(hr.sets).map((s) => ({
        label: DS[s] || s,
        values: [hr.sets[s].convnext_only.accuracy, hr.sets[s].zero_shot_only.accuracy, hr.sets[s].probe_only.accuracy, hr.sets[s].hybrid.accuracy] })),
        ["ConvNeXt only", "SigLIP zero-shot", "SigLIP + head", "Hybrid (ours)"], ["#b9b9ae", "#8dc63f", "#6c9a22", "#2f4a10"])}</div>
      <div class="table-wrap" style="margin-top:.8rem"><table class="table"><thead><tr><th>Dataset</th><th>n</th><th>Hybrid acc.</th><th>Macro F1</th><th>Answered</th><th>Acc. when answered</th></tr></thead><tbody>
        ${Object.entries(hr.sets).map(([s, v]) => `<tr><td>${DS[s] || s}${v.held_out ? "" : " <span class='amber-pill'>tuning</span>"}</td><td>${v.hybrid.n}</td><td><b>${pct(v.hybrid.accuracy)}</b></td><td>${pct(v.hybrid.macro_f1)}</td><td>${pct(v.hybrid.answered)}</td><td>${pct(v.hybrid.accuracy_on_answered)}</td></tr>`).join("")}
      </tbody></table></div>
    </div></section>` : ""}

  <section class="sec px">
    <div class="sec-head"><h2 class="section-title">Dataset & documented split</h2><span class="see-all">seed ${sp.seed}</span></div>
    <div class="card" style="padding:1rem">
      <p class="muted" style="font-size:.88rem">Garbage Classification (12 classes, Kaggle). Stratified 70% train / 15% validation / 15% test. The test split was used once, for the final numbers. Every file and its split: <code>artifacts/split.csv</code>.</p>
      <div class="table-wrap" style="margin-top:.6rem"><table class="table"><thead><tr><th>Class</th><th>Bin</th><th>Train</th><th>Val</th><th>Test</th><th>Total</th></tr></thead><tbody>
      ${Object.entries(sp.per_class).map(([c, v]) => `<tr><td>${esc(m.class_display[c])}</td><td>${esc(SHORT[v.category])}</td><td>${v.train}</td><td>${v.val}</td><td>${v.test}</td><td>${v.total}</td></tr>`).join("")}
      <tr><td><b>Total</b></td><td></td><td><b>${sp.split_totals.train}</b></td><td><b>${sp.split_totals.val}</b></td><td><b>${sp.split_totals.test}</b></td><td><b>${sp.total_images}</b></td></tr>
      </tbody></table></div>
    </div>
  </section>

  <section class="sec px">
    <div class="sec-head"><h2 class="section-title">ConvNeXt test-set detail</h2></div>
    <div class="grid2">
      <figure class="card" style="margin:0;padding:.75rem"><img src="/media/confusion_matrix_5.png" alt="5-bin confusion matrix"><figcaption class="muted" style="font-size:.8rem;padding:.4rem">5 cafeteria bins</figcaption></figure>
      <figure class="card" style="margin:0;padding:.75rem"><img src="/media/confusion_matrix_12.png" alt="12-class confusion matrix"><figcaption class="muted" style="font-size:.8rem;padding:.4rem">12 classes</figcaption></figure>
    </div>
    <div class="card" style="padding:1rem;margin-top:1rem"><div class="table-wrap"><table class="table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr></thead><tbody>
      ${Object.entries(cr.fine_grained.per_class).map(([c, v]) => `<tr><td>${esc(m.class_display[c])}</td><td>${v.precision.toFixed(3)}</td><td>${v.recall.toFixed(3)}</td><td>${v["f1-score"].toFixed(3)}</td><td>${v.support}</td></tr>`).join("")}
    </tbody></table></div></div>
  </section>

  <section class="sec px">
    <div class="sec-head"><h2 class="section-title">Failure cases</h2><span class="see-all">most confident mistakes</span></div>
    <div class="gallery">${cr.gallery.failures.slice(0, 8).map((f) => `
      <figure class="card"><img loading="lazy" src="/gallery/${f.file}" alt=""><figcaption><b>${esc(m.class_display[f.true])}</b> → ${esc(m.class_display[f.pred])} (${pct(f.confidence, 0)})<br>
      <span class="muted">${f.same_bin ? "same bin anyway" : f.flagged ? "caught by the gate" : "wrong bin"}</span></figcaption></figure>`).join("")}</div>
  </section>

  ${m.training_log ? `<section class="sec px">
    <div class="sec-head"><h2 class="section-title">Training</h2><span class="see-all">ConvNeXt-Tiny · RTX 3050</span></div>
    <div class="card" style="padding:1rem"><div class="chart">${lineChart(m.training_log.map((e) => e.epoch), [
      { name: "Val accuracy", color: "#2f4a10", values: m.training_log.map((e) => e.val_accuracy) },
      { name: "Val macro-F1", color: "#e2791b", values: m.training_log.map((e) => e.val_macro_f1) }], { min: 0.9, max: 1, xlabel: "epoch" })}</div></div>
  </section>` : ""}`;
}

// ------------------------------------------------------------------ STRESS view
function viewStress() {
  const cr = S.meta.convnext_report, hr = S.meta.hybrid_report;
  const sl = (id, label, min, max, step, val, fmt = (v) => v) =>
    `<label class="slider"><span class="top"><span>${label}</span><output id="o-${id}">${fmt(val)}</output></span>
     <input type="range" id="s-${id}" min="${min}" max="${max}" step="${step}" value="${val}"></label>`;
  const T = { gamma: "Lighting γ (<1 dim · >1 overexposed)", occlusion_pct: "Occlusion (% of image hidden)", glare: "Overhead glare", blur: "Blur (px)" };
  return `
  <section class="sec px reveal">
    <p class="label">Lighting & occlusion</p><h1 style="font-size:1.75rem;margin-top:.2rem">Stress test</h1>
    <p class="muted" style="margin-top:.4rem;max-width:46rem">Cafeteria photos are dim, glary, half-covered by a hand. Degrade the image live and watch the answer - and the "not sure" safety net - react.</p>
  </section>
  <section class="sec px"><div class="grid2 grid2-wide">
    <div class="card scanner">
      <div class="drop" id="stress-img">${S.current ? `<img src="${S.current.url}" alt="">` : `<div class="empty"><div class="big-ic">${I.sun(30)}</div><b style="color:var(--ink)">Scan or pick an image first</b><br><span style="font-size:.88rem">or choose a sample below</span></div>`}</div>
      ${sl("gamma", "Lighting γ", 0.3, 2.5, 0.1, S.stress.gamma, (v) => Number(v).toFixed(1))}
      ${sl("glare", "Overhead glare", 0, 1, 0.1, S.stress.glare, (v) => Number(v).toFixed(1))}
      ${sl("occlusion", "Hand / napkin occlusion", 0, 70, 5, S.stress.occlusion, (v) => `${v}%`)}
      ${sl("blur", "Motion blur", 0, 10, 1, S.stress.blur)}
      <div style="display:flex;gap:.5rem;margin-top:1rem;flex-wrap:wrap">
        <button class="btn btn-ghost btn-sm" id="stress-reset">${I.refresh(16)} Reset</button>
        ${S.meta.demos.slice(0, 4).map((d) => `<button class="chip" data-sdemo="${esc(d.file)}">${esc(d.name)}</button>`).join("")}
      </div>
    </div>
    <div id="stress-result">${S.result && S.current ? resultCard(S.result) : resultPlaceholder()}</div>
  </div></section>

  ${hr && hr.robustness ? `<section class="sec px">
    <div class="sec-head"><h2 class="section-title">Measured: hybrid vs ConvNeXt</h2><span class="see-all">${hr.robustness.gamma ? "240 test images" : ""}</span></div>
    <div class="grid2">${Object.entries(hr.robustness).map(([k, rows]) => `<div class="card" style="padding:1rem"><p style="font-weight:700">${T[k]}</p>
      <div class="chart">${lineChart(rows.map((r) => r.value), [
        { name: "Hybrid", color: "#2f4a10", values: rows.map((r) => r.hybrid_accuracy) },
        { name: "ConvNeXt only", color: "#b9b9ae", values: rows.map((r) => r.convnext_accuracy) },
        { name: "Hybrid answered", color: "#e2791b", dash: true, values: rows.map((r) => r.hybrid_answered) }], { min: 0, max: 1 })}</div></div>`).join("")}</div>
  </section>` : ""}

  ${cr && cr.robustness ? `<section class="sec px">
    <div class="sec-head"><h2 class="section-title">Full sweep (ConvNeXt, 480 test images)</h2></div>
    <div class="grid2">${Object.entries(cr.robustness).map(([k, rows]) => `<div class="card" style="padding:1rem"><p style="font-weight:700">${T[k]}</p>
      <div class="chart">${lineChart(rows.map((r) => r.value), [
        { name: "Accuracy (12 classes)", color: "#2f4a10", values: rows.map((r) => r.accuracy_12) },
        { name: "Bin accuracy", color: "#6c9a22", values: rows.map((r) => r.accuracy_5) },
        { name: "Flagged 'not sure'", color: "#e2791b", dash: true, values: rows.map((r) => r.flag_rate) }], { min: 0, max: 1 })}</div></div>`).join("")}</div>
    <div class="card" style="padding:1.1rem;margin-top:1rem">
      <p style="font-weight:700">What we learned</p>
      <ul style="margin:.5rem 0 0;padding-left:1.1rem;font-size:.92rem;display:grid;gap:.35rem">
        <li><b>Lighting is handled well</b> - training-time colour jitter keeps accuracy above ~92% from very dim to overexposed.</li>
        <li><b>Occlusion is the weak spot</b> - once 40%+ of the item is hidden, accuracy falls fast; the gate flags more images as it does.</li>
        <li><b>Glare and blur</b> degrade slowly; strong glare hides labels and shape.</li>
        <li><b>Mitigations:</b> ColorJitter + RandomErasing in training (done), confidence/entropy gate with "retake / check label" (done), diffused LED ring light and fixed camera at the bin (proposed).</li>
      </ul></div>
  </section>` : ""}`;
}

let stressTimer = null;
function bindStress() {
  const upd = () => {
    S.stress = { gamma: +$("#s-gamma").value, glare: +$("#s-glare").value, occlusion: +$("#s-occlusion").value, blur: +$("#s-blur").value };
    $("#o-gamma").textContent = S.stress.gamma.toFixed(1);
    $("#o-glare").textContent = S.stress.glare.toFixed(1);
    $("#o-occlusion").textContent = `${S.stress.occlusion}%`;
    $("#o-blur").textContent = S.stress.blur;
    clearTimeout(stressTimer);
    stressTimer = setTimeout(runStress, 220);
  };
  ["gamma", "glare", "occlusion", "blur"].forEach((k) => ($(`#s-${k}`).oninput = upd));
  $("#stress-reset").onclick = () => { S.stress = { gamma: 1, glare: 0, occlusion: 0, blur: 0 }; render(); runStress(); };
  document.querySelectorAll("[data-sdemo]").forEach((b) => (b.onclick = async () => {
    const blob = await (await fetch(`/demo/${encodeURIComponent(b.dataset.sdemo)}`)).blob();
    if (S.current) URL.revokeObjectURL(S.current.url);
    S.current = { blob, url: URL.createObjectURL(blob) };
    runStress();
  }));
  bindResult();
}
async function runStress() {
  if (!S.current) return;
  try {
    const r = await classify(S.current.blob, S.stress);
    S.result = r;
    const box = $("#stress-img");
    if (box) box.innerHTML = `<img src="${r.preview || S.current.url}" alt="Degraded image">`;
    const out = $("#stress-result");
    if (out) { out.innerHTML = resultCard(r); bindResult(); }
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ ABOUT view
function viewAbout() {
  const hr = S.meta.hybrid_report;
  const cfg = S.meta.config;
  return `
  <section class="sec px reveal">
    <p class="label">How it works</p><h1 style="font-size:1.75rem;margin-top:.2rem">Three experts, one honest answer</h1>
    <p class="muted" style="margin-top:.4rem;max-width:46rem">A campus cafeteria produces paper, plastic, metal, food scraps and non-recyclables. One wrong item contaminates a whole bin and someone has to re-sort it by hand. Binzy tells students the right bin in under a second - and says so when it isn't sure.</p>
  </section>
  <section class="sec px"><div class="card" style="padding:1rem"><img src="/media/architecture.png" alt="System architecture diagram" style="border-radius:.75rem"></div></section>
  <section class="sec px"><div class="grid2 stagger">
    ${[
      ["leaf", "SigLIP zero-shot · open vocabulary", `Google's SigLIP (400M) compares the photo with ${hr ? hr.vocabulary.waste_items : "250"} text descriptions - "a chocolate bar wrapper", "a clay kulhad tea cup", "leftover rice" - plus "not waste" concepts like people, phones and scenery. It recognises items that were never in any training set.`],
      ["chart", "SigLIP + trained head", "A logistic-regression head on SigLIP image features, trained on our 70% training split. Learns the dataset's 12 classes with general-purpose features."],
      ["scan", "ConvNeXt-Tiny, fine-tuned", "ImageNet-pretrained CNN fine-tuned end-to-end on the training split with lighting (ColorJitter) and occlusion (RandomErasing) augmentation. 98% on in-domain photos."],
      ["shield", "Safety gate", `Fused 12-class probabilities are summed into the 5 bins. If "not waste" probability > ${cfg.ood_threshold.toFixed(2)} → out of scope. If bin confidence < ${cfg.conf_threshold.toFixed(2)} or normalised entropy > ${cfg.entropy_threshold.toFixed(2)} → "Not sure - check the label or ask staff". Thresholds are set from in-scope data only.`],
    ].map(([ic, t, d]) => `<div class="card" style="padding:1.1rem;display:flex;gap:.9rem"><span class="icon-tile">${I[ic](24)}</span><div><p style="font-weight:700">${t}</p><p class="muted" style="font-size:.9rem;margin-top:.25rem">${d}</p></div></div>`).join("")}
  </div></section>
  ${hr ? `<section class="sec px"><div class="card" style="padding:1.1rem">
    <p style="font-weight:700">Fusion weights (tuned on Kaggle validation + TrashNet)</p>
    <div class="bars" style="margin-top:.6rem">
      <div class="bar"><span>Zero-shot</span><span class="track"><span class="fill" style="width:${cfg.w_zs * 100}%;background:#8dc63f"></span></span><span class="v">${cfg.w_zs.toFixed(1)}</span></div>
      <div class="bar"><span>SigLIP head</span><span class="track"><span class="fill" style="width:${cfg.w_lp * 100}%;background:#6c9a22"></span></span><span class="v">${cfg.w_lp.toFixed(1)}</span></div>
      <div class="bar"><span>ConvNeXt</span><span class="track"><span class="fill" style="width:${cfg.w_cn * 100}%;background:#2f4a10"></span></span><span class="v">${cfg.w_cn.toFixed(1)}</span></div>
    </div></div></section>` : ""}

  <section class="sec px">
    <div class="sec-head"><h2 class="section-title">Responsible AI</h2></div>
    <div class="grid2 stagger">${[
      ["shield", "Privacy", "Photos are processed in memory on the local machine and discarded. Nothing is written to disk or sent to a cloud service. Session stats are counts only."],
      ["globe", "Bias & domain shift", "Training photos are mostly clean single items from outside India. The open-vocabulary expert adds Indian cafeteria items (kulhads, leaf plates, samosas, tetra packs); real-world accuracy is still expected to be lower than test scores."],
      ["help", "Human oversight", "Uncertain or out-of-scope photos never get a bin decision. Students check the label, ask staff, or pick the bin themselves. Batteries and broken glass always go to staff."],
      ["live", "Accessibility", "Icons plus text (never colour alone), plain-language steps, big touch targets, works on any phone browser, dark mode."],
    ].map(([ic, t, d]) => `<div class="card" style="padding:1.1rem;display:flex;gap:.9rem"><span class="icon-tile">${I[ic](24)}</span><div><p style="font-weight:700">${t}</p><p class="muted" style="font-size:.9rem;margin-top:.25rem">${d}</p></div></div>`).join("")}
    </div>
  </section>

  <section class="sec px">
    <div class="sec-head"><h2 class="section-title">UN Sustainable Development Goals</h2></div>
    <div class="row">${[
      ["#bf8b2e", "12.5", "Responsible consumption", "Cleaner recycling streams mean more paper, plastic and metal actually gets recovered."],
      ["#fd9d24", "11.6", "Sustainable cities", "Better campus waste management and less manual re-sorting by sanitation staff."],
      ["#3f7e44", "13", "Climate action", "Food scraps go to compost/biogas instead of landfill, cutting methane emissions."],
    ].map(([c, n, t, d]) => `<div class="card" style="width:min(300px,80vw);padding:1.1rem"><span style="display:inline-block;background:${c};color:#fff;font-weight:800;border-radius:.6rem;padding:.25rem .6rem">SDG ${n}</span>
      <p style="font-weight:700;margin-top:.6rem">${t}</p><p class="muted" style="font-size:.88rem;margin-top:.25rem">${d}</p></div>`).join("")}</div>
  </section>

  <section class="sec px"><div class="grid2">
    <div class="card" style="padding:1.1rem"><p class="label">Implemented today</p><ul style="margin:.5rem 0 0;padding-left:1.1rem;display:grid;gap:.3rem;font-size:.92rem">
      <li>Hybrid recogniser: SigLIP zero-shot + SigLIP head + fine-tuned ConvNeXt</li><li>Upload, camera capture and live auto-scan</li>
      <li>5 bins, item-specific disposal tips, out-of-scope + "not sure" safety net</li><li>Documented 70/15/15 split, test + external-dataset evaluation</li>
      <li>Measured lighting / occlusion / glare / blur robustness + live stress test</li></ul></div>
    <div class="card" style="padding:1.1rem"><p class="label">Next steps</p><ul style="margin:.5rem 0 0;padding-left:1.1rem;display:grid;gap:.3rem;font-size:.92rem">
      <li>Collect labelled photos in the VIT cafeteria and fine-tune on them</li><li>Multi-item tray detection (YOLO + TACO dataset)</li>
      <li>Edge box at the bin: ONNX / INT8, LED ring light, lid that opens the right bin</li><li>Contamination dashboard for the sustainability team</li></ul></div>
  </div></section>`;
}

// ------------------------------------------------------------------ charts (inline SVG)
function lineChart(xs, series, { min = 0, max = 1, xlabel = "" } = {}) {
  const W = 340, H = 170, L = 34, R = 10, T = 10, B = 26;
  const xi = (i) => L + (i / Math.max(1, xs.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - min) / (max - min)) * (H - T - B);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => { const v = min + f * (max - min);
    return `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 3}" font-size="9" text-anchor="end" fill="var(--muted)">${Math.round(v * 100)}%</text>`; }).join("");
  const xt = xs.map((x, i) => `<text x="${xi(i)}" y="${H - 8}" font-size="9" text-anchor="middle" fill="var(--muted)">${x}</text>`).join("");
  const lines = series.map((s) => {
    const pts = s.values.map((v, i) => (v == null ? null : `${xi(i)},${y(Math.max(min, Math.min(max, v)))}`)).filter(Boolean);
    return `<polyline points="${pts.join(" ")}" fill="none" stroke="${s.color}" stroke-width="2.4" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>` +
      s.values.map((v, i) => (v == null ? "" : `<circle cx="${xi(i)}" cy="${y(Math.max(min, Math.min(max, v)))}" r="2.6" fill="${s.color}"><title>${s.name}: ${pct(v)}</title></circle>`)).join("");
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${series.map((s) => s.name).join(", ")}">${grid}${xt}${lines}${xlabel ? `<text x="${W - R}" y="${H - 8}" font-size="9" text-anchor="end" fill="var(--muted)"></text>` : ""}</svg>
    <div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`;
}

function groupedBars(groups, names, colors) {
  const W = 360, H = 190, L = 34, B = 34, T = 14;
  const gw = (W - L - 6) / groups.length, bw = Math.min(18, (gw - 14) / names.length);
  const lo = 0.6;  // axis starts at 60% so differences between models are visible
  const y = (v) => T + (1 - (Math.max(lo, v) - lo) / (1 - lo)) * (H - T - B);
  const grid = [0.6, 0.7, 0.8, 0.9, 1].map((v) => `<line x1="${L}" x2="${W - 6}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 3}" font-size="9" text-anchor="end" fill="var(--muted)">${Math.round(v * 100)}%</text>`).join("");
  const bars = groups.map((g, gi) => {
    const x0 = L + gi * gw + (gw - bw * names.length) / 2;
    return g.values.map((v, i) => `<rect x="${x0 + i * bw}" y="${y(v)}" width="${bw - 2}" height="${H - B - y(v)}" rx="3" fill="${colors[i]}"><title>${names[i]}: ${pct(v)}</title></rect>`).join("") +
      `<text x="${L + gi * gw + gw / 2}" y="${H - 18}" font-size="9" text-anchor="middle" fill="var(--ink-2)" font-weight="700">${esc(g.label)}</text>` +
      `<text x="${x0 + (names.length - 1) * bw + bw / 2 - 1}" y="${y(g.values[names.length - 1]) - 4}" font-size="9" text-anchor="middle" fill="${colors[names.length - 1]}" font-weight="800">${(g.values[names.length - 1] * 100).toFixed(1)}%</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by dataset">${grid}${bars}</svg>
    <div class="legend">${names.map((n, i) => `<span><i style="background:${colors[i]}"></i>${n}</span>`).join("")}</div>`;
}

// ------------------------------------------------------------------ settings
function openSettings() {
  const cfg = { ...S.meta.config, ...S.overrides };
  const night = document.documentElement.dataset.theme === "night";
  const { root } = sheet("Settings", `
    <p class="label">Safety gate</p>
    <label class="slider"><span class="top"><span>Minimum bin confidence</span><output id="o-c">${cfg.conf_threshold.toFixed(2)}</output></span><input type="range" id="t-c" min="0.3" max="0.95" step="0.05" value="${cfg.conf_threshold}"></label>
    <label class="slider"><span class="top"><span>Maximum uncertainty (entropy)</span><output id="o-e">${cfg.entropy_threshold.toFixed(2)}</output></span><input type="range" id="t-e" min="0.2" max="0.95" step="0.05" value="${cfg.entropy_threshold}"></label>
    <label class="slider"><span class="top"><span>"Not waste" cut-off</span><output id="o-o">${cfg.ood_threshold.toFixed(2)}</output></span><input type="range" id="t-o" min="0.1" max="0.95" step="0.05" value="${cfg.ood_threshold}"></label>
    <div style="display:flex;gap:.5rem;margin-top:1.1rem;flex-wrap:wrap">
      <button class="btn btn-ghost btn-sm" id="t-reset">${I.refresh(16)} Default thresholds</button>
      <button class="btn btn-ghost btn-sm" id="t-theme">${I.moon(16)} ${night ? "Light" : "Night"} mode</button>
    </div>
    <p class="muted" style="font-size:.8rem;margin-top:1rem">Running on ${esc(S.meta.device.toUpperCase())}. Defaults were chosen so 97% of in-scope validation photos are answered.</p>`);
  const bind = (id, key, out) => ($(`#${id}`, root).oninput = (e) => { S.overrides[key] = +e.target.value; $(`#${out}`, root).textContent = (+e.target.value).toFixed(2); });
  bind("t-c", "conf_threshold", "o-c"); bind("t-e", "entropy_threshold", "o-e"); bind("t-o", "ood_threshold", "o-o");
  $("#t-reset", root).onclick = () => { S.overrides = {}; $("#sheet-root").innerHTML = ""; openSettings(); };
  $("#t-theme", root).onclick = () => {
    const t = document.documentElement.dataset.theme === "night" ? "" : "night";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("binzy-theme", t); } catch { /* ignore */ }
    $("#sheet-root").innerHTML = ""; openSettings();
  };
}

// ------------------------------------------------------------------ router
function route() { return (location.hash.replace(/^#\//, "") || "scan").split("?")[0]; }
function render() {
  const r = route();
  if (r !== "scan" && S.live.stream) stopCamera();
  renderNav(r);
  const v = $("#view");
  if (r === "insights") v.innerHTML = viewInsights();
  else if (r === "stress") { v.innerHTML = viewStress(); bindStress(); }
  else if (r === "about") v.innerHTML = viewAbout();
  else { v.innerHTML = viewScan(); bindScan(); }
}
window.addEventListener("hashchange", () => { render(); window.scrollTo(0, 0); });

["file-input", "file-capture"].forEach((id) => ($(`#${id}`).onchange = (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  if (route() === "stress") {
    if (S.current) URL.revokeObjectURL(S.current.url);
    S.current = { blob: f, url: URL.createObjectURL(f) }; runStress();
  } else { stopCamera(); render(); handleBlob(f); }
}));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData || {}).items || []].find((i) => i.type.startsWith("image/"));
  if (item && route() === "scan") { stopCamera(); handleBlob(item.getAsFile()); }
});

(async function boot() {
  shell();
  $("#view").innerHTML = `<div style="display:grid;place-items:center;min-height:60vh;gap:1rem;text-align:center">${logoMark(56)}<div class="spinner dark"></div><p class="muted">Waking up the models…</p></div>`;
  for (let i = 0; i < 120; i++) {
    try { S.meta = await api("/api/meta"); break; } catch { await new Promise((r) => setTimeout(r, 1000)); }
  }
  if (!S.meta) { $("#view").innerHTML = `<p class="px">Server not reachable.</p>`; return; }
  render();
})();
