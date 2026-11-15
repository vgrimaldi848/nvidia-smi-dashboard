#!/usr/bin/env python3
"""Single-file Flask app: a lightweight multi-GPU monitor fed by nvidia-smi.

Run it, then open  http://127.0.0.1:8081  (no authentication). Stop with CTRL+C.
Only third-party dependency:  pip install flask

Validated design notes
----------------------
* One non-interactive call per sample:
    nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits
  `power.draw` / `fan.speed` legitimately return "[N/A]" on many cards -> parsed to None.
* The subprocess is never executed on a request thread: the collector runs in its own
  daemon thread and executes `subprocess.run(timeout=...)` inside a ThreadPoolExecutor,
  so the HTTP process can never block on nvidia-smi.
* FIFO history == collections.deque(maxlen=100): appending the 101st sample evicts the oldest.
* Graceful CTRL+C: SIGINT/SIGTERM handlers are installed on the main thread (signals only work
  there); they set a threading.Event which releases the main thread, which then stops the
  collector and calls server.shutdown()/server_close(). Reloader is disabled deliberately.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import subprocess
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server

# ----------------------------------------------------------------- requirements
HOST = "127.0.0.1"        # fixed by the spec
PORT = 8081               # fixed by the spec
MAX_POINTS = 100          # FIFO depth per metric (older records are purged automatically)
DEFAULT_INTERVAL_SEC = 2.0
RATE_OPTIONS = [1, 2, 5, 15, 60]          # seconds offered by the top-right selector
MIN_INTERVAL_SEC, MAX_INTERVAL_SEC = 0.2, 900.0
CMD_TIMEOUT_SEC = 5.0

# nvidia-smi is assumed to be installed; override only for testing with a fake binary.
SMI_BIN = os.environ.get("NVIDIA_SMI_BIN", "nvidia-smi")

QUERY_FIELDS = (
    "index", "name", "pci.bus_id",                    # card model + PCIe slot
    "temperature.gpu", "fan.speed",                   # temperature, fan (% or N/A)
    "power.draw", "power.limit",                      # power / max power
    "memory.used", "memory.free",                     # memory used / available
    "memory.total", "utilization.gpu",                # capacity (info only)
)
QUERY_ARG = "--query-gpu=" + ",".join(QUERY_FIELDS)
FORMAT_ARG = "--format=csv,noheader,nounits"

METRIC_KEYS = ("temperature", "util_gpu", "fan", "power_draw", "power_limit", "mem_used_mib", "mem_free_mib")
# how the per-card metrics are aggregated into the "all cards" section
AGG_MODES = OrderedDict(
    (("temperature", "max"), ("util_gpu", "avg"), ("fan", "max"), ("power_draw", "sum"),
     ("power_limit", "sum"), ("mem_used_mib", "sum"), ("mem_free_mib", "sum"))
)


# ------------------------------------------------------------------- nvidia-smi
def smi_command(binary: str = SMI_BIN) -> list[str]:
    return [str(binary), QUERY_ARG, FORMAT_ARG]


def to_float(raw):
    """nvidia-smi returns '[N/A]' / 'N/A' for unsupported sensors -> None."""
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def parse_smi_output(text: str) -> list[dict]:
    readings: list[dict] = []
    lines = [line for line in text.splitlines() if line.strip()]
    for row in csv.reader(lines):                     # csv module: quoting-safe
        cells = [cell.strip() for cell in row]
        if len(cells) < len(QUERY_FIELDS):
            continue
        raw = dict(zip(QUERY_FIELDS, cells))
        readings.append({
            "index": raw["index"] or str(len(readings)),
            "name": raw["name"],
            "pci_bus_id": raw["pci.bus_id"],
            "memory_total_mib": to_float(raw["memory.total"]),
            "temperature": to_float(raw["temperature.gpu"]),
            "util_gpu": to_float(raw["utilization.gpu"]),   # <-- ADD  (None on vGPU/MIG where it is [N/A])
            "fan": to_float(raw["fan.speed"]),          # None -> fanless / passive cooling
            "power_draw": to_float(raw["power.draw"]),
            "power_limit": to_float(raw["power.limit"]),
            "mem_used_mib": to_float(raw["memory.used"]),
            "mem_free_mib": to_float(raw["memory.free"]),
        })
    return readings


def run_smi_query(binary: str, timeout: float = CMD_TIMEOUT_SEC):
    """Executed in a worker thread -> never blocks the Flask/HTTP process."""
    completed = subprocess.run(
        smi_command(binary), shell=False, capture_output=True, text=True, timeout=timeout, check=False
    )
    return completed.returncode, completed.stdout, completed.stderr


# ---------------------------------------------------------------- data retention
class Store:
    """Thread-safe in-memory ring buffers; maxlen enforces the 100-point FIFO purge."""

    def __init__(self, maxlen: int = MAX_POINTS) -> None:
        self.maxlen = max(1, int(maxlen))
        self._lock = threading.Lock()
        self.times: deque = deque(maxlen=self.maxlen)
        self.gpus: "OrderedDict[str, dict]" = OrderedDict()
        self.agg: dict = {key: deque(maxlen=self.maxlen) for key in AGG_MODES}
        self.samples = 0
        self.error = None

    def _discover(self, index: str) -> dict:
        entry = {
            "info": {"index": index, "name": "unknown", "pci_bus_id": "unknown",
                     "memory_total_mib": None},
            "hist": {key: deque(maxlen=self.maxlen) for key in METRIC_KEYS},
        }
        self.gpus[index] = entry
        return entry

    def record(self, ts: float, readings: list[dict]) -> None:
        by_index: "OrderedDict[str, dict]" = OrderedDict()
        for reading in readings:
            by_index.setdefault(str(reading.get("index", "0")), reading)
        with self._lock:
            for index, reading in by_index.items():
                entry = self.gpus.get(index) or self._discover(index)
                entry["info"]["name"] = reading.get("name") or entry["info"]["name"]
                entry["info"]["pci_bus_id"] = reading.get("pci_bus_id") or entry["info"]["pci_bus_id"]
                total = reading.get("memory_total_mib")
                if total is not None:
                    entry["info"]["memory_total_mib"] = total
            # a card that disappeared stops being tracked (keeps the view in sync)
            for gone in [key for key in self.gpus if key not in by_index]:
                del self.gpus[gone]

            self.times.append(ts)
            for index, entry in self.gpus.items():
                reading = by_index.get(index)          # may be None -> keeps columns aligned
                for key in METRIC_KEYS:
                    entry["hist"][key].append(None if reading is None else reading[key])

            for key, mode in AGG_MODES.items():
                values = [rec[key] for rec in readings if rec.get(key) is not None]
                if not values:
                    self.agg[key].append(None)
                elif mode == "max":
                    self.agg[key].append(max(values))
                elif mode == "avg":
                    self.agg[key].append(sum(values) / float(len(values)))
                else:
                    self.agg[key].append(sum(values))

            self.samples += 1
            self.error = None

    def record_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def snapshot(self) -> dict:
        with self._lock:
            gpus = []
            for entry in self.gpus.values():
                gpu = dict(entry["info"])
                for key, ring in entry["hist"].items():
                    gpu[key] = [None if val is None else round(float(val), 2) for val in ring]
                gpus.append(gpu)
            aggregate = {key: [None if val is None else round(float(val), 2) for val in ring]
                         for key, ring in self.agg.items()}
            return {
                "points": self.maxlen,
                "samples": self.samples,
                "error": self.error,
                "times": [round(t, 3) for t in self.times],
                "count": len(gpus),
                "aggregate": aggregate,
                "gpus": gpus,
            }


# -------------------------------------------------------------------- collector
class Collector(threading.Thread):
    """Polls nvidia-smi off the main process; sampling interval is settable live."""

    def __init__(self, store: Store, interval_sec: float, stop_event: threading.Event,
                 binary: str = SMI_BIN) -> None:
        super().__init__(name="nvidia-smi-collector", daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.binary = binary
        self._live = True
        self._interval = clamp_interval(interval_sec)
        self._wake = threading.Event()                 # lets a live change take effect at once
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="smi")

    # -- live configuration -------------------------------------------------
    @property
    def interval(self) -> float:
        return self._interval

    @property
    def live(self) -> bool:
        return self._live

    def set_interval(self, value) -> None:
        self._interval = clamp_interval(value)
        self._wake.set()

    def set_live(self, value: bool) -> None:
        self._live = bool(value)
        self._wake.set()

    def stop(self) -> None:
        self.stop_event.set()
        self._wake.set()

    def shutdown_pool(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)   # py>=3.9
        except TypeError:                                          # older python
            self._pool.shutdown(wait=False)

    # -- loop ---------------------------------------------------------------
    def run(self) -> None:
        logging.info("collector started (every %.1fs, %d points per metric)",
                     self._interval, self.store.maxlen)
        while not self.stop_event.is_set():
            self._wake.clear()
            if self._live:
                self.sample_once()
            timeout = None if not self._live else float(self._interval)
            self._wake.wait(timeout)                   # interruptible sleep -> CTRL+C is instant
        logging.info("collector stopped after %d samples", self.store.samples)

    def sample_once(self) -> None:
        try:
            future = self._pool.submit(run_smi_query, self.binary, CMD_TIMEOUT_SEC)
            code, out, err = future.result(timeout=CMD_TIMEOUT_SEC + 3.0)
        except Exception as exc:                       # missing binary, timeout, ...
            self.store.record_error("%s: %s" % (type(exc).__name__, exc))
            return
        if code != 0:
            lines = (err or out or "").strip().splitlines()
            detail = " | ".join(lines[:3]) if lines else ""
            self.store.record_error("nvidia-smi exit code %s: %s" % (code, detail))
            return
        readings = parse_smi_output(out)
        if not readings:
            self.store.record_error("nvidia-smi returned no GPU rows")
            return
        self.store.record(time.time(), readings)


def clamp_interval(value) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_SEC
    if seconds != seconds:                              # NaN guard
        return DEFAULT_INTERVAL_SEC
    return max(MIN_INTERVAL_SEC, min(MAX_INTERVAL_SEC, seconds))


# ------------------------------------------------------------------- web layer
def create_app(store: Store, collector: Collector) -> Flask:
    app = Flask(__name__)

    def payload() -> dict:
        data = store.snapshot()
        data["interval"] = collector.interval
        data["live"] = collector.live
        data["rate_options"] = list(RATE_OPTIONS)
        data["source"] = SMI_BIN
        return data

    @app.route("/", methods=["GET"])
    def index():
        return Response(HTML_PAGE, mimetype="text/html; charset=utf-8")

    @app.route("/api/status", methods=["GET"])
    def api_status():
        return jsonify(payload())

    @app.route("/api/config", methods=["GET"])
    def api_config_get():
        return jsonify({"interval": collector.interval, "live": collector.live,
                        "points": store.maxlen, "rate_options": list(RATE_OPTIONS)})

    @app.route("/api/config", methods=["POST", "PATCH"])
    def api_config_set():
        body = request.get_json(silent=True) or {}
        if "interval" in body:
            collector.set_interval(body.get("interval"))
        if "live" in body:
            collector.set_live(bool(body.get("live")))
        collector.wake() if hasattr(collector, "wake") else None
        return jsonify({"interval": collector.interval, "live": collector.live,
                        "points": store.maxlen, "rate_options": list(RATE_OPTIONS)})

    @app.route("/favicon.ico", methods=["GET"])
    def favicon():
        return Response(status=204)

    return app


HTML_PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GPU monitor</title>
<style>
:root{--bg:#0b0f17;--panel:#121a26;--line:#1e2a3d;--txt:#dbe4f0;--muted:#8aa0bb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.bar{position:sticky;top:0;z-index:3;display:flex;gap:12px;align-items:center;justify-content:space-between;padding:10px 16px;background:#0d131e;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:8px;font-weight:650}
.top-meta{color:var(--muted);font-weight:400;font-size:12.5px;margin-left:10px}
.controls{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted)}
select{background:#0b1220;color:var(--txt);border:1px solid #29405e;border-radius:7px;padding:6px 8px}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 7px #22c55eaa}
.dot.off{background:#f59e0b;box-shadow:none}
.switch{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;font-size:12.5px;color:var(--muted)}
.switch input{position:absolute;opacity:0;width:0;height:0}
.slider{width:36px;height:20px;border-radius:99px;background:#31415a;position:relative;display:inline-block;transition:.15s ease}
.slider i{position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:#fff;transition:.15s ease}
.switch input:checked+.slider{background:#1f9d63}
.switch input:checked+.slider i{transform:translateX(16px)}
main{padding:14px 16px 22px;display:flex;flex-direction:column;gap:16px}
h2{margin:0 0 8px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
.err{padding:8px;border-left:3px solid #ef4444;background:#2a1518;color:#fda4af;border-radius:6px;font-size:13px;white-space:pre-wrap;margin:0}
.hidden{display:none}
#gpuList{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(420px,1fr))}
@media (max-width:900px){#gpuList{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px 8px;min-width:0}
.card.idle{opacity:.45;border-style:dashed}
.chead{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.ttl{font-weight:650;font-size:14px}
.sub{color:var(--muted);font-size:12px;margin-top:3px;font-family:ui-monospace,SFMono-Regular,monospace;word-break:break-all}
.tiles{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.tile{border:1px solid var(--line);background:#0e1522;border-radius:9px;padding:6px 7px;min-width:0}
.tlabel{display:flex;justify-content:space-between;gap:6px;font-size:11px;color:var(--muted)}
.tval{color:var(--txt);font-weight:650;white-space:nowrap}
canvas{width:100%;height:104px;display:block}
footer{color:var(--muted);font-size:12px;padding:0 16px 8px;margin-top:auto}
kbd{background:#1b2537;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-family:ui-monospace,monospace}
</style>
</head>
<body>
<header class="bar">
  <div class="brand"><span class="dot" id="liveDot"></span><span>GPU monitor</span>
    <span class="top-meta" id="metaTop">connecting to the local collector ...</span></div>
  <div class="controls">
    <label for="rateSel">collect every</label>
    <select id="rateSel" title="nvidia-smi collection frequency"></select>
    <label class="switch" title="Live = keep collecting and refreshing, Idle = freeze the page">
      <input type="checkbox" id="liveChk" checked/>
      <span class="slider"><i></i></span><span id="liveTxt">Live</span>
    </label>
  </div>
</header>

<main>
  <section><h2>All cards &mdash; aggregate</h2><div class="tiles" id="aggTiles"></div></section>
  <section><h2>Dedicated per-card sections (switch a card off to put it idle)</h2>
          <div id="gpuList"></div></section>
  <p id="errBox" class="err hidden"></p>
</main>
<footer>Rolling window: the last <b>100</b> samples per metric (FIFO) &middot; source: nvidia-smi
 &middot; press <kbd>CTRL+C</kbd> in the terminal to stop the application.</footer>

<script>
"use strict";
/* metric descriptors: one rolling diagram per metric, plus its aggregate twin */
var METRICS = [
  { key:"temperature",  label:"Temperature (degC)", unit:" degC", color:"#fb7185" },
  { key:"util_gpu",     label:"GPU usage (%)",      unit:" %",    color:"#e879f9", zeroBase:true, util:true },
  { key:"fan",          label:"Fan speed (%)",      unit:" %",    color:"#22c55e", zeroBase:true, fan:true },
  { key:"power_draw",   label:"Power / max power",  unit:" W",    color:"#fbbf24", zeroBase:true,
    extra:{ key:"power_limit", label:"max", color:"#6b86ad" } },
  { key:"mem_free_mib", label:"Memory available",   unit:" MiB",  color:"#38bdf8" },
  { key:"mem_used_mib", label:"Memory used",        unit:" MiB",  color:"#a78bfa" }
];

var state = { points:100, intervalSec:2, live:true, sig:"", timer:null, busy:false };
var tilesAgg = {};       /* metric key -> tile (aggregate section) */
var cards    = {};       /* gpu index  -> { info, grid, tiles }   */
var idle     = {};       /* gpu index  -> true when the card switch is off */

function $(id){ return document.getElementById(id); }
function isNum(v){ return typeof v === "number" && !isNaN(v); }
function fmt(v){ if(!isNum(v)){ return "n/a"; }
  var d = (Math.abs(v) >= 1000) ? 0 : 1;
  return v.toLocaleString("en-US",{ maximumFractionDigits:d }); }
function last(arr){ return (!arr || arr.length === 0) ? null : arr[arr.length - 1]; }
function clock(ts){ if(!isNum(ts)){ return ""; }
  var d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US",{ hour12:false, hour:"2-digit", minute:"2-digit", second:"2-digit" }); }

/* ---------- rolling chart: auto-scaled y axis, newest sample pinned right ------- */
function drawChart(cv, seriesList, times){
  cv._series = seriesList; cv._times = times || []; paint(cv);
}
function paint(cv){
  var series = cv._series || [], times = cv._times || [];
  var pts = Math.max(2, state.points);
  var dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  var w = cv.clientWidth || 300, h = cv.clientHeight || 104;
  var pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if(cv.width !== pw || cv.height !== ph){ cv.width = pw; cv.height = ph; }
  var ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  var padL = 46, padR = 10, padT = 10, padB = 20;
  var plotW = Math.max(20, w - padL - padR), plotH = Math.max(20, h - padT - padB);

  ctx.strokeStyle = "#1c2739"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8aa0bb"; ctx.font = "10px ui-monospace,SFMono-Regular,monospace";
  ctx.textAlign = "left";
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(w - padR, padT + plotH); ctx.stroke();

  var lo = Infinity, hi = -Infinity, any = false;
  for(var s = 0; s < series.length; s++){
    var data = series[s].data || [];
    for(var j = 0; j < data.length; j++){ var v = data[j]; if(isNum(v)){ any = true; lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  }
  if(!any){ ctx.fillText("waiting for samples", padL + 4, h / 2); return; }
  var span = (hi - lo) || Math.abs(hi) || 1;
  hi = hi + span * 0.15; lo = lo - span * 0.15;

  function yAt(v){ return padT + plotH * (1 - (v - lo) / (hi - lo)); }

  ctx.textAlign = "right";
  var marks = [hi, (hi + lo) / 2, lo];
  for(var m = 0; m < marks.length; m++){
    var yv = marks[m], yy = yAt(yv);
    ctx.strokeStyle = "#172131"; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
    ctx.fillStyle = "#6d84a3"; ctx.fillText(fmt(yv), padL - 4, yy + 3);
  }

  var count = times.length || 1;
  var dx = plotW / Math.max(1, pts - 1);
  var offset = Math.max(0, pts - count);
  function xAt(j){ return padL + (offset + j) * dx; }

  for(var k = 0; k < series.length; k++){
    var sr = series[k], arr = sr.data || [];
    var len = Math.min(arr.length, count); if(len === 0){ continue; }
    ctx.strokeStyle = sr.color; ctx.lineWidth = 1.6; ctx.beginPath();
    var started = false;
    for(var i = 0; i < len; i++){
      var val = arr[i];
      if(!isNum(val)){ started = false; continue; }        /* gap on N/A samples */
      var x = xAt(i), y = yAt(val);
      if(!started){ ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();
    for(var q = len - 1; q >= 0; q--){
      if(isNum(arr[q])){ ctx.fillStyle = sr.color; ctx.beginPath();
        ctx.arc(xAt(q), yAt(arr[q]), 2.4, 0, Math.PI * 2); ctx.fill(); break; }
    }
  }
  ctx.textAlign = "left"; ctx.fillStyle = "#6d84a3";
  ctx.fillText(clock(times[0]) + "  ->  now", padL, h - 5);
}

/* ---------- tiles / cards ------------------------------------------------------ */
function buildTile(metric, container){
  var root = document.createElement("div"); root.className = "tile";
  var head = document.createElement("div"); head.className = "tlabel";
  var name = document.createElement("span"); name.textContent = metric.label;
  var val  = document.createElement("b");   val.className = "tval"; val.textContent = "-";
  head.appendChild(name); head.appendChild(val);
  var cv = document.createElement("canvas"); cv.width = 600; cv.height = 220;
  root.appendChild(head); root.appendChild(cv); container.appendChild(root);
  return { metric:metric, canvas:cv, valueEl:val };
}
function seriesOf(tile, obj){
  var m = tile.metric;
  var out = [{ label:m.label, color:m.color, data:(obj && obj[m.key]) || [] }];
  if(m.extra){ out.push({ label:m.extra.label, color:m.extra.color, data:(obj && obj[m.extra.key]) || [] }); }
  return out;
}
function valueText(tile, obj){
  var m = tile.metric, v = last(obj ? obj[m.key] : null);
  if(m.fan){ if(!isNum(v)){ return "n/a (no sensor)"; } return v <= 0 ? "OFF / idle" : fmt(v) + " %"; }
  if(m.extra){ return fmt(v) + " / " + fmt(last(obj ? obj[m.extra.key] : null)) + m.unit; }
  return fmt(v) + m.unit;
}
function tileUpdate(tile, obj, times){
  tile.valueEl.textContent = valueText(tile, obj);
  drawChart(tile.canvas, seriesOf(tile, obj), times);
}
function createCard(gpu){
  var idx = String(gpu.index);
  var card = document.createElement("article"); card.className = "card";
  var head = document.createElement("div"); head.className = "chead";
  var box = document.createElement("div");
  var ttl = document.createElement("div"); ttl.className = "ttl"; ttl.textContent = "GPU " + idx;
  var sub = document.createElement("div"); sub.className = "sub"; sub.textContent = "-";
  box.appendChild(ttl); box.appendChild(sub);

  var swLabel = document.createElement("label"); swLabel.className = "switch";
  var input = document.createElement("input"); input.type = "checkbox";
  input.checked = !idle[idx];
  var slider = document.createElement("span"); slider.className = "slider";
  slider.appendChild(document.createElement("i"));
  var txt = document.createElement("span"); txt.textContent = idle[idx] ? "Idle" : "Active";
  swLabel.appendChild(input); swLabel.appendChild(slider); swLabel.appendChild(txt);

  var grid = document.createElement("div"); grid.className = "tiles";
  var entry = { root:card, info:sub, grid:grid, tiles:{} };
  for(var i = 0; i < METRICS.length; i++){
    entry.tiles[METRICS[i].key] = buildTile(METRICS[i], grid);
  }
  input.addEventListener("change", function(){
    idle[idx] = !input.checked;
    entry.root.classList.toggle("idle", !input.checked);
    entry.grid.style.display = input.checked ? "" : "none";   /* idle card: charts not rendered */
    txt.textContent = input.checked ? "Active" : "Idle";
  });

  head.appendChild(box); head.appendChild(swLabel);
  card.appendChild(head); card.appendChild(grid);
  grid.style.display = idle[idx] ? "none" : "";
  if(idle[idx]){ card.classList.add("idle"); }
  cards[idx] = entry;
  $("gpuList").appendChild(card);
}
function applySig(p){
  var list = p.gpus || [], sig = [];
  for(var i = 0; i < list.length; i++){ sig.push(String(list[i].index) + ":" + (list[i].name || "")); }
  var joined = sig.join("|");
  if(joined === state.sig){ return; }
  state.sig = joined; cards = {}; $("gpuList").textContent = "";
  for(var g = 0; g < list.length; g++){ createCard(list[g]); }
}

/* ---------- render + polling --------------------------------------------------- */
function render(p){
  if(!p){ return; }
  state.points = p.points || state.points;
  if(typeof p.interval === "number"){ state.intervalSec = p.interval; }
  state.live = (p.live !== false);

  var sel = $("rateSel");
  if(sel.options.length === 0 && Array.isArray(p.rate_options)){
    for(var o = 0; o < p.rate_options.length; o++){
      var opt = document.createElement("option");
      opt.value = String(Number(p.rate_options[o]));
      opt.textContent = String(Number(p.rate_options[o])) + " s";
      sel.appendChild(opt);
    }
  }
  if(sel.options.length){ sel.value = String(Number(state.intervalSec)); if(sel.selectedIndex < 0){ sel.selectedIndex = 0; } }

  $("liveChk").checked = state.live;
  $("liveTxt").textContent = state.live ? "Live" : "Idle";
  $("liveDot").className = state.live ? "dot" : "dot off";

  var err = $("errBox");
  if(p.error){ err.textContent = "collector: " + p.error; err.classList.remove("hidden"); }
  else { err.classList.add("hidden"); }

  $("metaTop").textContent = (p.count || 0) + " card(s) - ring buffer " + state.points +
    " samples/metric - every " + String(state.intervalSec) + "s";

  var times = p.times || [], agg = p.aggregate || {};
  for(var i = 0; i < METRICS.length; i++){ tileUpdate(tilesAgg[METRICS[i].key], agg, times); }
  var gpus = p.gpus || [];
  for(var g = 0; g < gpus.length; g++){
    var gpu = gpus[g], entry = cards[String(gpu.index)];
    if(!entry){ continue; }
    entry.info.textContent = (gpu.name || "unknown") + " - slot " + (gpu.pci_bus_id || "?") +
      " - " + fmt(gpu.memory_total_mib) + " MiB total";
    for(var m = 0; m < METRICS.length; m++){ tileUpdate(entry.tiles[METRICS[m].key], gpu, times); }
  }
}
function schedule(){
  if(state.timer){ clearTimeout(state.timer); state.timer = null; }
  if(!state.live){ return; }
  state.timer = setTimeout(tick, Math.max(250, Math.round((state.intervalSec || 2) * 1000)));
}
function tick(){
  if(state.busy){ schedule(); return; }
  state.busy = true;
  fetch("/api/status", { cache:"no-store" })
    .then(function(r){ if(!r.ok){ throw new Error("HTTP " + r.status); } return r.json(); })
    .then(function(p){ applySig(p); render(p); })
    .catch(function(e){ var box = $("errBox"); box.textContent = "UI: cannot reach the local app (" + e.message + ")"; box.classList.remove("hidden"); })
    .then(function(){ state.busy = false; schedule(); });
}
function patchConfig(body){
  fetch("/api/config", { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body) })
    .then(function(r){ return r.json(); })
    .then(function(p){ if(!p || typeof p !== "object"){ return; }
        state.live = (p.live !== false);
        if(typeof p.interval === "number"){ state.intervalSec = p.interval; }
        render({ points:p.points || state.points, interval:p.interval, live:p.live,
                 rate_options:p.rate_options, count:(state.lastCount || 0), times:[], aggregate:{}, gpus:[] , error:null });
        if(state.live){ tick(); } })
    .catch(function(e){ var box = $("errBox"); box.textContent = "UI: config failed (" + e.message + ")"; box.classList.remove("hidden"); });
}

/* boot */
for(var t = 0; t < METRICS.length; t++){ tilesAgg[METRICS[t].key] = buildTile(METRICS[t], $("aggTiles")); }
$("rateSel").addEventListener("change", function(){ patchConfig({ interval: Number(this.value) }); });
$("liveChk").addEventListener("change", function(){ patchConfig({ live: this.checked === true }); });
window.addEventListener("resize", function(){ if(state.sig !== ""){ tick(); } });
tick();
</script>
</body>
</html>'''


# ------------------------------------------------------------------- entrypoint
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="nvidia-smi multi-GPU monitor (single page Flask app)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC,
                        help="initial collection period in seconds (also changeable in the page)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log HTTP requests too")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.ERROR if not args.verbose else logging.INFO)

    stop_event = threading.Event()
    store = Store(MAX_POINTS)
    collector = Collector(store, args.interval, stop_event, SMI_BIN)
    app = create_app(store, collector)

    # own server handle => clean shutdown (the reloader stays disabled on purpose)
    server = make_server(HOST, PORT, app, threaded=True)
    serving = threading.Thread(target=server.serve_forever, name="http", daemon=True)

    def on_signal(signum, _frame):
        logging.shutdown()
        stop_event.set()
        try:
            print("\nreceived %s, shutting down gently ..." % signal.Signals(signum).name, flush=True)
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)          # signals must be handled on the main thread
        except ValueError:
            pass                                  # not the main thread: keep default behaviour

    serving.start()
    collector.start()
    print("GPU monitor ready  ->  http://%s:%d  (CTRL+C to stop)" % (HOST, PORT), flush=True)

    try:
        stop_event.wait()                          # main thread blocks here until CTRL+C/SIGTERM
    finally:
        collector.stop()
        collector.shutdown_pool()
        server.shutdown()                           # stops accepting/handling requests
        server.server_close()
        serving.join(timeout=5.0)
        print("stopped cleanly.", flush=True)
    return 0


if __name__ == "__main__":
   raise SystemExit(main())
