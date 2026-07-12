// Polls /api/nodes + /api/summary every 2s and re-renders.
const POLL_MS = 2000;

const $ = (id) => document.getElementById(id);

function fmtAge(s) {
  if (s == null || isNaN(s)) return "-";
  if (s < 90) return s.toFixed(0) + "s ago";
  if (s < 3600) return (s / 60).toFixed(1) + "m ago";
  return (s / 3600).toFixed(1) + "h ago";
}
function fmtUptime(s) {
  if (!s) return "-";
  if (s < 60)    return s.toFixed(0) + "s";
  if (s < 3600)  return (s / 60).toFixed(1) + "m";
  if (s < 86400) return (s / 3600).toFixed(1) + "h";
  return (s / 86400).toFixed(1) + "d";
}
function fmtNum(n, digits = 0) {
  if (n == null || isNaN(n)) return "-";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
function fmtPct(p) {
  if (p == null || isNaN(p)) return "-";
  return (p * 100).toFixed(1) + "%";
}

function dot(online, age) {
  if (!online) return '<span class="dot bad"></span>';
  if (age > 30) return '<span class="dot warn"></span>';
  return '<span class="dot ok"></span>';
}

function renderSummary(s) {
  const cards = [
    { label: "Edge online",    val: s.edges_online,  unit: s.edges_offline ? `+${s.edges_offline} off` : "" },
    { label: "Cloud online",   val: s.clouds_online, unit: s.clouds_offline ? `+${s.clouds_offline} off` : "" },
    { label: "Total requests", val: fmtNum(s.total_requests) },
    { label: "Routed to cloud",val: fmtNum(s.total_routed) },
    { label: "Global offload", val: fmtPct(s.global_offload_rate) },
    { label: "Total bandwidth",val: fmtNum(s.total_bandwidth_kb / 1024, 2), unit: "MB" },
  ];
  $("summary").innerHTML = cards.map(c => `
    <div class="card">
      <div class="label">${c.label}</div>
      <div class="val">${c.val}<span class="unit">${c.unit || ""}</span></div>
    </div>`).join("");
}

function renderNode(n) {
  const role = n.role || "?";
  const online = n.online;
  const stats = n.stats || {};
  const cls = ["node", role];
  if (!online) cls.push("offline");

  const kvs = [];
  kvs.push(["status", `${dot(online, n.last_seen_age_s)}${online ? "online" : "offline"}`]);
  kvs.push(["host:port", `${n.hostname || "?"}:${n.port || "?"}`]);
  kvs.push(["last seen", fmtAge(n.last_seen_age_s)]);
  kvs.push(["uptime", fmtUptime(stats.uptime_s)]);
  kvs.push(["requests", fmtNum(stats.requests_total)]);
  if (role === "edge") {
    kvs.push(["routed", `${fmtNum(stats.requests_routed)} (${fmtPct(stats.requests_total ? stats.requests_routed / stats.requests_total : null)})`]);
    kvs.push(["avg margin",   stats.avg_margin == null ? "-" : stats.avg_margin.toFixed(3)]);
    kvs.push(["avg bandwidth",stats.avg_bandwidth_kb == null ? "-" : stats.avg_bandwidth_kb.toFixed(2) + " KB"]);
    kvs.push(["bw total",     fmtNum(stats.bandwidth_total_kb, 1) + " KB"]);
  } else if (role === "cloud") {
    kvs.push(["avg classify", stats.avg_classify_ms == null ? "-" : stats.avg_classify_ms.toFixed(0) + " ms"]);
  }
  kvs.push(["avg latency", stats.avg_latency_ms == null ? "-" : stats.avg_latency_ms.toFixed(0) + " ms"]);
  if (stats.last_label) kvs.push(["last label", stats.last_label]);

  const cfg = n.config || {};
  const cfgKeys = Object.keys(cfg);
  if (cfgKeys.length) {
    for (const k of cfgKeys) kvs.push([k, cfg[k]]);
  }

  return `
    <div class="${cls.join(" ")}">
      <div class="head">
        <span class="id">${n.node_id}</span>
        <span class="role">${role}</span>
      </div>
      <dl class="kv">
        ${kvs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}
      </dl>
    </div>`;
}

function partition(nodes) {
  const edges = [], clouds = [], offline = [];
  for (const n of Object.values(nodes)) {
    if (!n.online) offline.push(n);
    else if (n.role === "edge") edges.push(n);
    else if (n.role === "cloud") clouds.push(n);
    else offline.push(n);
  }
  const sortIt = (a, b) => a.node_id.localeCompare(b.node_id);
  return [edges.sort(sortIt), clouds.sort(sortIt), offline.sort(sortIt)];
}

function render(nodes, summary) {
  renderSummary(summary);
  const [edges, clouds, offline] = partition(nodes);
  $("edge-count").textContent  = edges.length  ? `(${edges.length})`  : "";
  $("cloud-count").textContent = clouds.length ? `(${clouds.length})` : "";
  $("off-count").textContent   = offline.length? `(${offline.length})`: "";
  $("edges").innerHTML  = edges.length  ? edges.map(renderNode).join("")  : '<div class="empty">no edge nodes connected yet</div>';
  $("clouds").innerHTML = clouds.length ? clouds.map(renderNode).join("") : '<div class="empty">no cloud nodes connected yet</div>';
  $("offline").innerHTML = offline.length ? offline.map(renderNode).join("") : '<div class="empty">no offline nodes</div>';
  $("last-update").textContent = `last update: ${new Date().toLocaleTimeString()}`;
}

async function poll() {
  try {
    const [n, s] = await Promise.all([
      fetch("/api/nodes").then(r => r.json()),
      fetch("/api/summary").then(r => r.json()),
    ]);
    render(n, s);
  } catch (e) {
    $("last-update").textContent = `error: ${e.message || e}`;
  }
}

poll();
setInterval(poll, POLL_MS);
