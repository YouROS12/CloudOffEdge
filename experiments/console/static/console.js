// Edge–cloud operations console. Polls /api/state and /api/nodes; renders with
// plain DOM + inline SVG (no dependencies, works offline on the farm network).
const FARM_VARS = { A: "--s1", B: "--s2", C: "--s3", D: "--s4" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const farmColor = (f) => css(FARM_VARS[f] || "--muted");
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");
const fmtPct = (x, d = 1) => (x == null ? "—" : (100 * x).toFixed(d) + " %");
const fmtMB = (kb) => (kb / 1024).toFixed(kb >= 10240 ? 0 : 1);
const fmtMs = (ms) => (ms == null ? "—" : ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function svg(tag, attrs = {}, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}

function renderClock(s) {
  const t = s.clock.sim_time;
  if (!t) { $("clock").textContent = "waiting for run…"; return; }
  const d = new Date(t);
  const day = Math.floor(s.clock.sim_hour_index / 24);
  $("clock").textContent = `Day ${day + 1} · ${DAYS[(d.getDay() + 6) % 7]} ${String(d.getHours()).padStart(2, "0")}:00`;
  $("weekFill").style.width = `${Math.min(100, ((s.clock.sim_hour_index + 1) / (s.meta.days * 24)) * 100)}%`;
}

function renderKpis(s) {
  const t = s.totals, L = s.latency;
  const tiles = [
    { label: "Images diagnosed", value: fmtInt(t.n), small: `/ ${fmtInt(s.meta.n_planned)}`,
      foot: `${fmtInt(t.n - t.esc)} answered on the farm`, meter: s.meta.n_planned ? t.n / s.meta.n_planned : 0 },
    { label: "Escalated to cloud", value: fmtPct(t.esc_rate), foot: `${fmtInt(t.esc)} images uploaded` },
    { label: "Uplink used", value: `${fmtMB(t.up_kb)}`, small: "MB",
      foot: t.reduction ? `vs ${fmtMB(t.cloud_only_kb)} MB always-cloud · <b>${t.reduction.toFixed(1)}× less</b>` : "—" },
    { label: "Delivered accuracy", value: fmtPct(t.acc), foot: `edge alone ${fmtPct(t.edge_acc)} · replay ground truth` },
    { label: "Answer on the farm", value: fmtMs(L.answer_p50), small: "median",
      foot: `cloud ${fmtMs(L.cloud_p50)}/image · confirmed ≤ ${L.delay_h_max != null ? L.delay_h_max.toFixed(0) : "—"} h later` },
  ];
  $("kpis").innerHTML = tiles.map((k) => `
    <div class="kpi"><div class="label">${k.label}</div>
      <div class="value num">${k.value}${k.small ? `<small>${k.small}</small>` : ""}</div>
      <div class="foot">${k.foot}</div>
      ${k.meter != null ? `<div class="meter"><i style="width:${(k.meter * 100).toFixed(1)}%"></i></div>` : ""}
    </div>`).join("");
}

function renderTimeline(s) {
  const farms = Object.keys(s.farms), H = s.hourly.length;
  const W = 900, Ht = 180, pl = 30, pr = 6, pt = 8, pb = 24;
  const iw = W - pl - pr, ih = Ht - pt - pb, bw = iw / H;
  const max = Math.max(4, ...s.hourly.map((h) => farms.reduce((a, f) => a + (h[f] || 0), 0)));
  const yMax = Math.ceil(max / 4) * 4;
  const root = svg("svg", { viewBox: `0 0 ${W} ${Ht}`, role: "img", "aria-label": "Escalations per simulated hour" });
  for (let d = 0; d < H / 24; d++) {  // night shading 19:00-07:00
    root.append(svg("rect", { x: pl + (d * 24 - 5) * bw, y: pt, width: 12 * bw, height: ih, fill: css("--night") }));
  }
  for (let i = 0; i <= 4; i++) {
    const y = pt + ih - (i / 4) * ih;
    root.append(svg("line", { x1: pl, x2: W - pr, y1: y, y2: y, stroke: css("--grid"), "stroke-width": 1 }));
    root.append(svg("text", { x: pl - 6, y: y + 3.5, "text-anchor": "end", "font-size": 10, fill: css("--muted") }, (yMax * i) / 4));
  }
  const now = s.clock.sim_hour_index;
  s.hourly.forEach((h, i) => {
    let y = pt + ih;
    farms.forEach((f) => {
      const v = h[f] || 0;
      if (!v) return;
      const hh = (v / yMax) * ih;
      root.append(svg("rect", { x: pl + i * bw + 0.6, y: y - hh, width: Math.max(1, bw - 1.2), height: Math.max(0, hh - 1),
                                rx: 1, fill: farmColor(f) }));
      y -= hh;
    });
  });
  for (let d = 0; d <= H / 24; d++) {
    const x = pl + d * 24 * bw;
    root.append(svg("line", { x1: x, x2: x, y1: pt, y2: pt + ih, stroke: css("--axis"), "stroke-width": d % 7 ? 0.6 : 1 }));
    if (d < H / 24) root.append(svg("text", { x: x + 12 * bw, y: Ht - 7, "text-anchor": "middle", "font-size": 10.5, fill: css("--ink2") }, `Day ${d + 1}`));
  }
  if (now > 0) root.append(svg("line", { x1: pl + (now + 1) * bw, x2: pl + (now + 1) * bw, y1: pt, y2: pt + ih,
                                         stroke: css("--ink"), "stroke-width": 1.2 }));
  $("timeline").replaceChildren(root);
  $("legend").innerHTML = farms.map((f) => `<span><i style="background:${farmColor(f)}"></i>Farm ${f}</span>`).join("")
    + `<span><i style="background:${css("--night")};box-shadow:0 0 0 1px var(--ring)"></i>night</span>`;
}

function renderFarms(s) {
  $("farms").innerHTML = Object.entries(s.farms).map(([f, v]) => {
    const rate = v.n ? v.esc / v.n : 0, red = v.up_kb ? v.cloud_only_kb / v.up_kb : null;
    const hours = v.hourly, mx = Math.max(1, ...hours), w = 220, h = 30;
    const pts = hours.map((x, i) => `${(i / (hours.length - 1)) * w},${h - (x / mx) * (h - 2) - 1}`).join(" ");
    const crops = Array.isArray(v.crops) ? v.crops.join(", ") : v.crops;
    return `<div class="farm">
      <div class="farm-head"><span class="badge" style="background:${farmColor(f)}">${f}</span>
        <span class="farm-type">${esc(v.type)}</span></div>
      <div class="crops">${esc(crops)}</div>
      <dl class="stats">
        <dt>diagnosed</dt><dd>${fmtInt(v.n)} / ${fmtInt(v.planned)}</dd>
        <dt>escalated</dt><dd>${fmtPct(rate)}</dd>
        <dt>uplink</dt><dd>${fmtMB(v.up_kb)} MB</dd>
        <dt>vs always-cloud</dt><dd>${red ? red.toFixed(1) + "×" : "—"}</dd>
        <dt>accuracy</dt><dd>${v.n ? fmtPct(v.correct / v.n) : "—"}</dd>
      </dl>
      <svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="hourly escalations">
        <polyline points="${pts}" fill="none" stroke="${farmColor(f)}" stroke-width="1.5"/></svg>
    </div>`;
  }).join("");
}

function renderTopology(s, nodes) {
  const W = 420, H = 230, farms = Object.keys(s.farms);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "System topology" });
  const ring = css("--ring"), ink2 = css("--ink2"), muted = css("--muted");
  root.append(svg("rect", { x: 8, y: 14, width: 150, height: 200, rx: 10, fill: "none", stroke: css("--axis"), "stroke-dasharray": "4 3" }));
  root.append(svg("text", { x: 83, y: 32, "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, fill: css("--ink") }, "Raspberry Pi 5"));
  root.append(svg("text", { x: 83, y: 46, "text-anchor": "middle", "font-size": 10, fill: muted }, "4 edge processes"));
  farms.forEach((f, i) => {
    const y = 66 + i * 36, on = (nodes || []).find((n) => n.id === `farm-${f}`)?.online !== false;
    root.append(svg("rect", { x: 22, y, width: 122, height: 28, rx: 7, fill: css("--surface"), stroke: ring }));
    root.append(svg("circle", { cx: 38, cy: y + 14, r: 8, fill: farmColor(f) }));
    root.append(svg("text", { x: 38, y: y + 17.5, "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: "#fff" }, f));
    root.append(svg("text", { x: 52, y: y + 18, "font-size": 10.5, fill: ink2 }, `${fmtInt(s.farms[f].n)} imgs`));
    root.append(svg("circle", { cx: 134, cy: y + 14, r: 3.5, fill: on ? css("--good") : css("--critical") }));
  });
  const cloud = (nodes || []).find((n) => n.role === "cloud");
  root.append(svg("rect", { x: 262, y: 62, width: 150, height: 104, rx: 10, fill: css("--s1-soft"), stroke: ring }));
  root.append(svg("text", { x: 337, y: 84, "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, fill: css("--ink") }, "Jetson Nano 4 GB"));
  ["DINOv2-L · DINOv3-L · CLIP-L", `TensorRT · ${s.flushes.n} batch${s.flushes.n === 1 ? "" : "es"}`,
   cloud ? (cloud.online ? "online" : "unreachable") : "—"].forEach((t, i) =>
    root.append(svg("text", { x: 337, y: 104 + i * 16, "text-anchor": "middle", "font-size": 10, fill: i === 2 && cloud && !cloud.online ? css("--critical") : ink2 }, t)));
  const defs = svg("defs", {}), marker = svg("marker", { id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5,
                                                         markerWidth: 7, markerHeight: 7, orient: "auto" });
  marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", fill: css("--s1") }));
  defs.append(marker);
  root.append(defs);
  root.append(svg("line", { x1: 160, y1: 114, x2: 258, y2: 114, stroke: css("--s1"), "stroke-width": 2, "marker-end": "url(#arr)" }));
  root.append(svg("text", { x: 209, y: 104, "text-anchor": "middle", "font-size": 10.5, "font-weight": 600, fill: css("--s1") }, `${fmtInt(s.totals.esc)} escalated`));
  root.append(svg("text", { x: 209, y: 130, "text-anchor": "middle", "font-size": 10, fill: muted }, `${fmtMB(s.totals.up_kb)} MB uplink`));
  root.append(svg("text", { x: 337, y: 190, "text-anchor": "middle", "font-size": 10, fill: muted }, "WireGuard overlay (Tailscale)"));
  $("topology").replaceChildren(root);
  $("topoNote").textContent = `τ = ${s.recent[0]?.threshold ?? "—"}`;
}

function renderFeed(s) {
  const start = new Date(s.meta.sim_start || s.recent[0]?.sim_time || 0);
  $("feed").innerHTML = s.recent.map((r) => {
    const routed = r.route_decision === "1", ok = r.final_correct === "1", m = Number(r.edge_margin), tau = Number(r.threshold);
    const t = new Date(r.sim_time), day = Math.floor((t - start) / 864e5) + 1;
    return `<div class="item">
      <img loading="lazy" src="/img?path=${encodeURIComponent(r.image)}" alt="">
      <div>
        <div class="t1">${esc(r.final_label)}</div>
        <div class="t2">Farm ${r.farm} · Day ${day} ${String(t.getHours()).padStart(2, "0")}:00 · margin ${m.toFixed(2)}${
          routed && r.cloud_ok === "1" && Number(r.answer_delay_h) > 0 ? ` · cloud +${r.answer_delay_h} h` : ""}</div>
        <div class="mbar"><i style="width:${Math.min(100, m * 100)}%"></i><b style="left:${tau * 100}%"></b></div>
      </div>
      <div style="display:grid;gap:4px;justify-items:end">
        <span class="route ${routed && r.cloud_ok === "1" ? "cloud" : "local"}"
          title="${routed && r.cloud_ok !== "1" ? "escalation failed; edge answer kept" : ""}">${
          routed ? (r.cloud_ok === "1" ? "CLOUD" : "FALLBACK") : "LOCAL"}</span>
        <span class="verdict ${ok ? "ok" : "bad"}">${ok ? "✓ correct" : "✗ " + esc(r.true_label)}</span>
      </div></div>`;
  }).join("");
}

function renderLatency(s) {
  const L = s.latency, W = 900, H = 96, pl = 130, pr = 70;
  const edge = L.edge_p50 || 0, net = L.net_p50 || 0, cl = L.cloud_p50 || 0;
  const max = Math.max(edge + net + cl, L.local_p50 || 0, 1);
  const x = (v) => pl + (v / max) * (W - pl - pr);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Latency per stage" });
  const rows = [["answer on the farm", [[L.answer_p50 || 0, css("--muted")]]],
                ["cloud confirmation", [[edge, css("--muted")], [net, css("--s4")], [cl, css("--s1")]]]];
  rows.forEach(([label, segs], i) => {
    const y = 14 + i * 40;
    root.append(svg("text", { x: pl - 10, y: y + 13, "text-anchor": "end", "font-size": 11.5, fill: css("--ink2") }, label));
    let acc = 0;
    segs.forEach(([v, c]) => {
      if (v <= 0) return;
      root.append(svg("rect", { x: x(acc) + (acc ? 1 : 0), y, width: Math.max(2, x(acc + v) - x(acc) - (acc ? 1 : 0)), height: 18, rx: 3, fill: c }));
      acc += v;
    });
    root.append(svg("text", { x: x(acc) + 8, y: y + 13, "font-size": 11.5, "font-weight": 600, fill: css("--ink") }, fmtMs(acc)));
  });
  const key = [["edge (Pi 5)", css("--muted"), edge], ["network", css("--s4"), net], ["cloud (Nano)", css("--s1"), cl]];
  key.forEach(([t, c, v], i) => {
    const kx = pl + i * 190;
    root.append(svg("rect", { x: kx, y: H - 12, width: 10, height: 10, rx: 2, fill: c }));
    root.append(svg("text", { x: kx + 15, y: H - 3, "font-size": 11, fill: css("--ink2") }, `${t} ${fmtMs(v || null)}`));
  });
  $("latency").replaceChildren(root);
}

function renderNodes(nodes) {
  $("nodes").innerHTML = nodes.map((n) => `<span class="node ${n.online ? "" : "off"}" title="${esc(n.detail)}">
    <span class="dot"></span>${esc(n.id)} <span class="role">${n.role}</span></span>`).join("");
}

let lastNodes = [];
async function tick() {
  try {
    const s = await (await fetch("/api/state")).json();
    renderClock(s); renderKpis(s); renderTimeline(s); renderFarms(s); renderTopology(s, lastNodes);
    renderFeed(s); renderLatency(s);
    $("disclosure").textContent = s.meta.disclosure || "";
  } catch (e) { console.error(e); }
}
async function nodesTick() {
  try { lastNodes = await (await fetch("/api/nodes")).json(); renderNodes(lastNodes); } catch (e) { console.error(e); }
}
nodesTick(); tick();
setInterval(tick, 1500);
setInterval(nodesTick, 5000);
