// The browser talks to the FastAPI backend through these JSON endpoints (same origin, so no CORS):
//   GET  /api/meta            -> airports, model info
//   GET  /api/destinations    -> where can you fly from an airport
//   GET  /api/carriers        -> which airlines fly that route
//   POST /api/predict         -> probability + reasons

const $ = (id) => document.getElementById(id);
const el = {
  origin: $("origin"), dest: $("dest"), carrier: $("carrier"), date: $("date"), time: $("time"),
  submit: $("submit"), form: $("flight-form"), error: $("form-error"),
};
let airports = {};

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? "Please check the date and time you entered." : body.detail;
    throw new Error(detail || "Something went wrong. Please try again.");
  }
  return body;
}

function fill(select, items, placeholder, label) {
  select.innerHTML = "";
  select.append(new Option(placeholder, ""));
  items.forEach((i) => select.append(new Option(label(i), i.code)));
  select.disabled = items.length === 0;
}

function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }
function updateSubmit() { el.submit.disabled = !(el.origin.value && el.dest.value && el.carrier.value && el.date.value && el.time.value); }

// ---------- initial load ----------
async function init() {
  const today = new Date();
  const iso = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  el.date.min = iso(today);
  const t = new Date(today.getTime() + 7 * 864e5);
  el.date.value = iso(t);
  try {
    const meta = await api("/api/meta");
    meta.airports.forEach((a) => (airports[a.code] = a.city));
    fill(el.origin, meta.airports, "Choose departure airport", (a) => `${a.code} · ${a.city}`);
    const bits = [];
    if (meta.roc_auc) bits.push(`ROC-AUC ${meta.roc_auc.toFixed(2)} on ${meta.test_years ? meta.test_years.join(", ") : "held-out"} flights the model never saw`);
    bits.push(`Typical flight: ${(meta.base_rate * 100).toFixed(0)}% arrive late`);
    bits.push(`History from ${meta.stats_year}`);
    $("model-note").textContent = bits.join(". ") + ".";
    window.__baseRate = meta.base_rate;
  } catch (e) {
    showError("The prediction service is not reachable right now. Refresh the page to try again.");
  }
}

el.origin.addEventListener("change", async () => {
  fill(el.dest, [], "Choose a departure airport first", () => "");
  fill(el.carrier, [], "Choose a destination first", () => "");
  updateSubmit();
  if (!el.origin.value) return;
  try {
    const dests = await api(`/api/destinations?origin=${el.origin.value}`);
    fill(el.dest, dests, "Choose destination", (a) => `${a.code} · ${a.city}`);
  } catch (e) { showError(e.message); }
});

el.dest.addEventListener("change", async () => {
  fill(el.carrier, [], "Choose a destination first", () => "");
  updateSubmit();
  if (!el.dest.value) return;
  try {
    const cs = await api(`/api/carriers?origin=${el.origin.value}&dest=${el.dest.value}`);
    fill(el.carrier, cs, "Choose airline", (c) => c.name);
    if (cs.length === 1) { el.carrier.value = cs[0].code; }
    updateSubmit();
  } catch (e) { showError(e.message); }
});

[el.carrier, el.date, el.time].forEach((n) => n.addEventListener("input", updateSubmit));

// ---------- predict ----------
el.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError("");
  el.submit.classList.add("loading");
  el.submit.disabled = true;
  el.submit.textContent = "Checking...";
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        flight_date: el.date.value, dep_time: el.time.value,
        carrier: el.carrier.value, origin: el.origin.value, dest: el.dest.value,
      }),
    });
    render(result);
  } catch (e) {
    showError(e.message);
  } finally {
    el.submit.classList.remove("loading");
    el.submit.textContent = "Check delay risk";
    updateSubmit();
  }
});

const VERDICT = {
  low: "Less likely to be late than a typical flight.",
  medium: "About as likely to be late as a typical flight.",
  high: "More likely to be late than a typical flight.",
};

function render(r) {
  $("stub-empty").hidden = true;
  $("stub-result").hidden = false;
  $("details").hidden = false;

  $("r-origin").textContent = el.origin.value;
  $("r-dest").textContent = el.dest.value;
  const carrierName = el.carrier.options[el.carrier.selectedIndex].text;
  $("r-meta").textContent = `${carrierName}, ${el.date.value} at ${el.time.value}`;

  const pct = Math.round(r.probability * 100);
  $("r-prob").textContent = `${pct}%`;
  $("r-verdict").textContent = `${VERDICT[r.risk]} About ${Math.round(r.base_rate * 100)}% of flights arrive late.`;

  const stub = $("stub-result");
  stub.className = `risk-${r.risk}`;
  // Scale the meter to 0-60% so differences are visible (most flights sit between 10% and 40%).
  const MAX = 0.6;
  $("meter-typical").style.left = `${Math.min(r.base_rate / MAX, 1) * 100}%`;
  $("meter").setAttribute("aria-label", `${pct}% delay risk, typical flight is ${Math.round(r.base_rate * 100)}%`);
  requestAnimationFrame(() => { $("meter-fill").style.width = `${Math.min(r.probability / MAX, 1) * 100}%`; });

  // reasons
  const ul = $("reasons");
  ul.innerHTML = "";
  const maxW = Math.max(...r.reasons.map((x) => x.weight), 0.01);
  r.reasons.forEach((x) => {
    const li = document.createElement("li");
    li.className = x.direction;
    li.innerHTML = `<span class="txt"></span><span class="bar"><i></i></span><span class="tag"></span>`;
    li.querySelector(".txt").textContent = x.text;
    li.querySelector("i").style.width = `${(x.weight / maxW) * 100}%`;
    li.querySelector(".tag").textContent = x.direction === "up" ? "Raises the risk" : "Lowers the risk";
    ul.append(li);
  });

  drawChart(r);
  $("details").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
}

function drawChart(r) {
  const rates = r.context.hourly_origin_rates;
  const sel = r.context.selected_hour;
  $("chart-title").textContent = `Late arrivals by departure hour at ${el.origin.value}`;
  $("chart-sub").textContent = `Share of ${el.origin.value} departures that arrived 15+ minutes late in ${r.context.stats_year}. Your departure hour is highlighted.`;

  const W = 520, H = 220, padL = 34, padB = 26, padT = 10;
  const vals = rates.map((v) => (v == null ? 0 : v));
  const top = Math.max(0.4, Math.ceil(Math.max(...vals) * 10) / 10);
  const bw = (W - padL) / 24;
  const y = (v) => padT + (H - padT - padB) * (1 - v / top);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Bar chart of late-arrival rate for each departure hour">`;
  for (let t = 0; t <= top + 1e-9; t += 0.1) {
    svg += `<line x1="${padL}" x2="${W}" y1="${y(t)}" y2="${y(t)}" stroke="#dde4ea"/><text x="${padL - 6}" y="${y(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text>`;
  }
  rates.forEach((v, h) => {
    if (v == null) return;
    const x = padL + h * bw + 2;
    svg += `<rect x="${x}" y="${y(v)}" width="${bw - 4}" height="${H - padB - y(v)}" rx="2" fill="${h === sel ? "#c4372a" : "#9fb4c7"}"><title>${h}:00 - ${(v * 100).toFixed(0)}% late</title></rect>`;
  });
  [0, 6, 12, 18, 23].forEach((h) => { svg += `<text x="${padL + h * bw + bw / 2}" y="${H - 8}" text-anchor="${h === 23 ? "end" : "middle"}">${h}:00</text>`; });
  const base = r.base_rate;
  svg += `<line x1="${padL}" x2="${W}" y1="${y(base)}" y2="${y(base)}" stroke="#0f2b46" stroke-dasharray="4 4"/>`;
  svg += `</svg>`;
  $("chart").innerHTML = svg;
}

init();
