"""Interactive L2 backtest dashboard (stdlib HTTP server + Plotly via CDN).

    python scripts/l2_dashboard.py          # then open http://localhost:8765

Pick an asset from data/l2/*.npz, choose the snapshot window (mapped to NSE
session timestamps when the file carries a `ts` field), tweak strategy knobs,
and run. Signals (regime, drift, imbalance forward-fill) are recomputed strictly
inside the selected window -- nothing outside the slice leaks in. Fills stay at
t+1 through the L2 book exactly as in l2_strategy.backtest.

PnL is reported both gross (fee-free, "IMC Prosperity" style) and net of a real
NSE cost stack applied per fill leg on traded notional:

    equity intraday : STT 0.025% (sell), NSE txn 0.00297%, SEBI 0.0001%,
                      stamp 0.003% (buy), GST 18% on charges, optional brokerage
    options         : STT 0.1% (sell premium), NSE txn 0.03503%, SEBI 0.0001%,
                      stamp 0.003% (buy), GST 18% on charges, optional brokerage
    custom          : flat bps per side

Fees never alter decisions (same as the original: sizing/entries are fee-blind),
so one simulation yields both curves.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_lib"))
from l2_strategy import setup_arrays, backtest, WARM, LAG

DATA_DIR = "data"
PORT = 8765
MAX_PLOT_POINTS = 4000

_raw_cache: dict[str, dict] = {}
_setup_cache: dict[tuple, dict] = {}


# ============================ NSE fee model ============================
def fee_per_leg(notional, is_buy, kind, mode, custom_bps=2.5, brokerage_bps=0.0):
    """Rupee cost of one fill leg of `notional` = qty * price."""
    if mode == "none":
        return 0.0
    if mode == "custom":
        return (custom_bps + brokerage_bps) / 1e4 * notional
    if mode == "binance":                      # USDT-M futures taker, VIP0
        return 4.5 / 1e4 * notional
    if kind == "option":                       # premium turnover
        stt   = 0.0 if is_buy else 0.00100 * notional     # 0.100% sell
        txn   = 0.0003503 * notional
    else:                                       # equity intraday
        stt   = 0.0 if is_buy else 0.00025 * notional     # 0.025% sell
        txn   = 0.0000297 * notional
    stamp = 0.00003 * notional if is_buy else 0.0
    sebi  = 0.000001 * notional
    brok  = brokerage_bps / 1e4 * notional
    gst   = 0.18 * (brok + txn + sebi)
    return stt + txn + stamp + sebi + brok + gst


def instrument_kind(symbol):
    import re
    s = symbol.upper()
    if "USDT" in s or "USDC" in s:
        return "crypto-perp"
    return "option" if re.search(r"\d+(CE|PE)$", s) else "equity"


# ============================== data =================================
def load_raw(path):
    if path not in _raw_cache:
        raw = np.load(path, allow_pickle=True)
        d = {k: raw[k] for k in raw.files}
        d["symbol"] = str(d.get("symbol", os.path.basename(path)))
        _raw_cache[path] = d
    return _raw_cache[path]


def list_assets():
    out = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True)):
        r = load_raw(p)
        n = int(r["bids"].shape[0])
        a = dict(file=p, symbol=r["symbol"], n=n,
                 kind=instrument_kind(r["symbol"]))
        if "ts" in r:
            a["ts0"] = int(r["ts"][0]); a["ts1"] = int(r["ts"][-1])
        out.append(a)
    return out


def setup_slice(path, t0, t1):
    key = (path, t0, t1)
    if key in _setup_cache:
        return _setup_cache[key]
    r = load_raw(path)
    m = (r["tr_bookidx"] >= t0) & (r["tr_bookidx"] < t1)
    D = setup_arrays(r["bids"][t0:t1], r["asks"][t0:t1],
                     r["tr_price"][m], r["tr_qty"][m], r["tr_side"][m],
                     r["tr_bookidx"][m] - t0, r["symbol"])
    D["ts"] = r["ts"][t0:t1].astype("int64") if "ts" in r else None
    _setup_cache[key] = D
    if len(_setup_cache) > 12:                     # bound memory
        _setup_cache.pop(next(iter(_setup_cache)))
    return D


# ============================ run + payload ============================
def _metrics(tr, pnl, mid):
    tr = np.asarray(tr) if len(tr) else np.array([0.0])
    pts = tr.mean() / tr.std() if tr.std() > 0 else 0.0
    return dict(pnl=float(pnl), sharpe=float(pts), tstat=float(pts * np.sqrt(len(tr))),
                trades=int(len(tr)), win_rate=float(100 * np.mean(tr > 0)),
                edge_bps=float(tr.mean() / np.median(mid) * 1e4))


def run_api(cfg):
    t_start = time.time()
    path = cfg["file"]; t0 = int(cfg["t0"]); t1 = int(cfg["t1"])
    r = load_raw(path); n_total = int(r["bids"].shape[0])
    t0 = max(0, t0); t1 = min(n_total, t1)
    if t1 - t0 < WARM + 50:
        return {"error": f"window too short: need > {WARM + 50} snapshots "
                         f"(warmup {WARM}), got {t1 - t0}"}

    D = setup_slice(path, t0, t1)
    R = backtest(D, use_drift_confirm=bool(cfg.get("drift_confirm", True)),
                 thr=float(cfg.get("thr", 0.4)), hold=int(cfg.get("hold", 15)),
                 fee_bps=0.0, sizing=cfg.get("sizing", "tanh"), record=True)

    # ---- fees per fill leg -> net equity + net per-trade pnl ----
    kind = instrument_kind(D["symbol"])
    mode = cfg.get("fee_mode", "nse")
    cbps = float(cfg.get("fee_bps", 2.5)); bbps = float(cfg.get("brokerage_bps", 0.0))
    N = D["N"]
    fee_ts = np.zeros(N); leg_fees = []
    for (te, dq, dc) in R["fills"]:
        notional = abs(dc)
        f = fee_per_leg(notional, dq > 0, kind, mode, cbps, bbps)
        fee_ts[te:] += f; leg_fees.append(f)
    equity_g = R["equity"]
    equity_n = equity_g - fee_ts
    tr_g = R["trade_pnls"]
    tr_n = tr_g.copy()
    for i in range(min(len(tr_n), len(leg_fees) // 2)):
        tr_n[i] -= leg_fees[2 * i] + leg_fees[2 * i + 1]
    fees_total = float(sum(leg_fees))

    gross = _metrics(tr_g, equity_g[-1], D["mid"])
    net = _metrics(tr_n, equity_n[-1], D["mid"])

    # ---- regime segments (local index -> x units) ----
    ts = D["ts"]
    def X(i):
        i = min(int(i), N - 1)
        return int(ts[i]) if ts is not None else int(i + t0)
    segs = []; s0 = 0
    reg = D["reg"]
    for t in range(1, N + 1):
        if t == N or reg[t] != reg[s0]:
            segs.append([X(s0), X(t - 1), int(reg[s0])]); s0 = t

    # ---- downsampled series ----
    stride = max(1, N // MAX_PLOT_POINTS)
    idx = np.arange(0, N, stride)
    dd = equity_n - np.maximum.accumulate(equity_n)
    series = dict(
        x=[X(i) for i in idx],
        mid=np.round(D["mid"][idx], 4).tolist(),
        eq_gross=np.round(equity_g[idx], 4).tolist(),
        eq_net=np.round(equity_n[idx], 4).tolist(),
        position=np.round(R["position"][idx], 3).tolist(),
        imb=np.round(D["imb"][idx], 3).tolist(),
        drift=D["drift"][idx].astype(int).tolist(),
        dd=np.round(dd[idx], 4).tolist(),
    )
    entries = [dict(x=X(te), side=int(s), px=float(px)) for te, s, px in R["entries"]]
    exits = [dict(x=X(te), side=int(s), px=float(px), pnl=float(p))
             for te, s, px, p, et in R["exits"]]

    reg_pct = {str(k): float(100 * np.mean(reg == k)) for k in (1, -1, 0)}
    return dict(
        symbol=D["symbol"], kind=kind, n=N, t0=t0, t1=t1,
        has_ts=ts is not None, warm_x=X(min(WARM, N - 1)),
        invalid_pct=float(100 * np.mean(~D["valid"])),
        regime_pct=reg_pct, gross=gross, net=net, fees_total=fees_total,
        fee_mode=mode, series=series, entries=entries, exits=exits,
        tr_gross=np.round(tr_g, 4).tolist(), tr_net=np.round(tr_n, 4).tolist(),
        segments=segs, elapsed=time.time() - t_start,
    )


# ================================ HTTP =================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/assets":
            self._send(200, list_assets())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            cfg = json.loads(self.rfile.read(n))
            self._send(200, run_api(cfg))
        except Exception as e:  # surface errors to the UI
            self._send(200, {"error": f"{type(e).__name__}: {e}"})


# ================================ page =================================
PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NSE L2 Backtester</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 :root{--bg:#0f1117;--panel:#181c26;--edge:#2a3040;--fg:#dde3ee;--dim:#8b93a7;
       --green:#3fb950;--red:#f85149;--amber:#d29922;--blue:#58a6ff}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:13px/1.45 "Segoe UI",system-ui,sans-serif}
 #wrap{display:grid;grid-template-columns:290px 1fr;min-height:100vh}
 #side{background:var(--panel);border-right:1px solid var(--edge);padding:16px;
       position:sticky;top:0;height:100vh;overflow-y:auto;box-sizing:border-box}
 #main{padding:16px 20px;min-width:0}
 h1{font-size:16px;margin:0 0 14px} h1 span{color:var(--dim);font-weight:400}
 label{display:block;color:var(--dim);margin:10px 0 3px;font-size:11px;
       text-transform:uppercase;letter-spacing:.05em}
 select,input{width:100%;box-sizing:border-box;background:#0d1017;color:var(--fg);
       border:1px solid var(--edge);border-radius:6px;padding:6px 8px;font-size:13px}
 .row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
 .chk{display:flex;align-items:center;gap:8px;margin-top:10px}
 .chk input{width:auto}
 button{width:100%;margin-top:16px;padding:9px;border:0;border-radius:6px;
       background:var(--blue);color:#04121f;font-weight:600;font-size:14px;cursor:pointer}
 button:disabled{opacity:.5;cursor:wait}
 #status{margin-top:10px;color:var(--dim);font-size:12px;min-height:16px}
 #tslabel{color:var(--dim);font-size:11px;margin-top:4px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:10px;margin-bottom:14px}
 .card{background:var(--panel);border:1px solid var(--edge);border-radius:8px;padding:10px 12px}
 .card .t{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
 .card .v{font-size:19px;font-weight:600;margin-top:2px}
 .card .s{color:var(--dim);font-size:11px;margin-top:2px}
 .pos{color:var(--green)} .neg{color:var(--red)}
 .plot{background:var(--panel);border:1px solid var(--edge);border-radius:8px;
       margin-bottom:12px;padding:4px}
 .prow{display:grid;grid-template-columns:3fr 2fr;gap:12px}
 .note{color:var(--dim);font-size:11px;margin:4px 0 16px}
 fieldset{border:1px solid var(--edge);border-radius:6px;margin-top:14px;padding:8px 10px}
 legend{color:var(--dim);font-size:11px;text-transform:uppercase;padding:0 4px}
</style></head><body>
<div id="wrap">
<div id="side">
  <h1>NSE L2 Backtester <span>| regime-switching book strategy</span></h1>
  <label>Asset</label><select id="asset"></select>
  <label>Window (snapshot index)</label>
  <div class="row2"><input id="t0" type="number" value="0" min="0">
                    <input id="t1" type="number" value="10000"></div>
  <div id="tslabel"></div>
  <fieldset><legend>Strategy</legend>
    <label>Sizing</label>
    <select id="sizing"><option value="tanh" selected>tanh (conviction)</option>
      <option value="arctan">arctan</option><option value="binary">binary</option></select>
    <div class="row2">
      <div><label>Imb. threshold</label><input id="thr" type="number" value="0.4" step="0.05" min="0" max="0.95"></div>
      <div><label>Hold (ticks)</label><input id="hold" type="number" value="15" min="1"></div>
    </div>
    <div class="chk"><input id="drift" type="checkbox" checked><span>drift confirmation</span></div>
  </fieldset>
  <fieldset><legend>Fees (net curve)</legend>
    <label>Model</label>
    <select id="feemode">
      <option value="nse" selected>NSE cost stack (STT+txn+stamp+GST)</option>
      <option value="binance">Binance futures taker (4.5 bps/side)</option>
      <option value="custom">custom bps / side</option>
      <option value="none">none (Prosperity mode)</option>
    </select>
    <div class="row2">
      <div><label>Custom bps/side</label><input id="feebps" type="number" value="2.5" step="0.5"></div>
      <div><label>Brokerage bps</label><input id="brok" type="number" value="0" step="0.5"></div>
    </div>
  </fieldset>
  <button id="runbtn" onclick="run()">Run backtest</button>
  <div id="status"></div>
</div>
<div id="main">
  <div class="note">Gross = fee-free PnL (IMC-Prosperity style). Net = after the selected fee
  model, charged per fill leg on traded notional. Signals are recomputed inside the selected
  window only; entries fill at t+1 by crossing the recorded L2 book. Background: green=DIR
  (trending), red=MM (mean-reverting), gray=FLAT.</div>
  <div class="cards" id="cards"></div>
  <div class="plot" id="p_mid"></div>
  <div class="plot" id="p_eq"></div>
  <div class="plot" id="p_pos"></div>
  <div class="plot" id="p_imb"></div>
  <div class="plot" id="p_dd"></div>
  <div class="prow"><div class="plot" id="p_hist"></div><div class="plot" id="p_reg"></div></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
let assets=[];
const REGC={"1":"rgba(63,185,80,%o)","-1":"rgba(248,81,73,%o)","0":"rgba(139,147,167,%o)"};
const fmtTs=ms=>new Date(ms).toISOString().replace('T',' ').slice(0,19)+" IST";

async function init(){
  assets=await (await fetch('/api/assets')).json();
  $('asset').innerHTML=assets.map((a,i)=>
    `<option value="${i}">${a.symbol}  [${a.kind}]  (${a.n.toLocaleString()} snaps)</option>`).join('');
  $('asset').onchange=onAsset; onAsset();
}
function onAsset(){
  const a=assets[+$('asset').value];
  $('t0').value=0; $('t1').value=a.n; $('t1').max=a.n;
  $('tslabel').textContent=a.ts0?`${fmtTs(a.ts0)} → ${fmtTs(a.ts1)}`:'';
}
function card(t,v,cls,s){return `<div class="card"><div class="t">${t}</div>
  <div class="v ${cls||''}">${v}</div>${s?`<div class="s">${s}</div>`:''}</div>`}
function sgn(x,d=2){return (x>=0?'+':'')+x.toFixed(d)}

async function run(){
  const a=assets[+$('asset').value]; $('runbtn').disabled=true;
  $('status').textContent='running…';
  const cfg={file:a.file,t0:+$('t0').value,t1:+$('t1').value,sizing:$('sizing').value,
    thr:+$('thr').value,hold:+$('hold').value,drift_confirm:$('drift').checked,
    fee_mode:$('feemode').value,fee_bps:+$('feebps').value,brokerage_bps:+$('brok').value};
  let d;
  try{ d=await (await fetch('/api/run',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})).json(); }
  catch(e){ d={error:String(e)}; }
  $('runbtn').disabled=false;
  if(d.error){ $('status').textContent='⚠ '+d.error; return; }
  $('status').textContent=`done in ${d.elapsed.toFixed(1)}s — ${d.n.toLocaleString()} snaps`;
  render(d);
}

function render(d){
  // ts encodes IST wall-clock as epoch ms; render as naive strings so Plotly
  // does not shift them into the browser's timezone
  const X = d.has_ts ? (v=>new Date(v).toISOString().replace('T',' ').slice(0,19)) : (v=>v);
  const x = d.series.x.map(X);
  const g=d.gross,n=d.net,rp=d.regime_pct;
  $('cards').innerHTML=
    card('Gross PnL (₹/unit)',sgn(g.pnl),g.pnl>=0?'pos':'neg',
         `Sharpe ${sgn(g.sharpe,3)} · t ${sgn(g.tstat,1)} · edge ${sgn(g.edge_bps,2)}bps`)+
    card(`Net PnL (${d.fee_mode})`,sgn(n.pnl),n.pnl>=0?'pos':'neg',
         `Sharpe ${sgn(n.sharpe,3)} · t ${sgn(n.tstat,1)} · fees ₹${d.fees_total.toFixed(2)}`)+
    card('Trades',g.trades,'',`win gross ${g.win_rate.toFixed(1)}% / net ${n.win_rate.toFixed(1)}%`)+
    card('Regime split',`${rp['1'].toFixed(1)}% DIR`,'',
         `MM ${rp['-1'].toFixed(1)}% · FLAT ${rp['0'].toFixed(1)}%`)+
    card('Invalid books',d.invalid_pct.toFixed(1)+'%','',`${d.kind} · window ${d.t0}–${d.t1}`);

  const shapes=op=>d.segments.map(s=>({type:'rect',xref:'x',yref:'paper',
    x0:X(s[0]),x1:X(s[1]),y0:0,y1:1,line:{width:0},layer:'below',
    fillcolor:REGC[String(s[2])].replace('%o',s[2]==0?op*0.35:op)}));
  const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:'#dde3ee',size:11},margin:{l:55,r:15,t:34,b:30},
    xaxis:{gridcolor:'#232838'},yaxis:{gridcolor:'#232838'},showlegend:true,
    legend:{orientation:'h',y:1.12,bgcolor:'rgba(0,0,0,0)'}};
  const L=(t,extra)=>Object.assign({},base,{title:{text:t,font:{size:13}}},extra||{});
  const C={displayModeBar:false,responsive:true};

  const eB=d.entries.filter(e=>e.side>0), eS=d.entries.filter(e=>e.side<0);
  Plotly.react('p_mid',[
    {x,y:d.series.mid,type:'scattergl',mode:'lines',name:'mid',
     line:{color:'#58a6ff',width:1}},
    {x:eB.map(e=>X(e.x)),y:eB.map(e=>e.px),type:'scattergl',mode:'markers',
     name:`long entry (${eB.length})`,marker:{symbol:'triangle-up',size:9,color:'#3fb950',
     line:{color:'#0f1117',width:1}}},
    {x:eS.map(e=>X(e.x)),y:eS.map(e=>e.px),type:'scattergl',mode:'markers',
     name:`short entry (${eS.length})`,marker:{symbol:'triangle-down',size:9,color:'#f85149',
     line:{color:'#0f1117',width:1}}},
    {x:d.exits.map(e=>X(e.x)),y:d.exits.map(e=>e.px),type:'scattergl',mode:'markers',
     name:`exit (${d.exits.length})`,marker:{symbol:'x',size:7,color:'#dde3ee'},
     text:d.exits.map(e=>'trade pnl '+sgn(e.pnl)),hoverinfo:'text+x+y'}],
    L(`${d.symbol} — midprice + trades (regime-shaded)`,
      {shapes:shapes(0.16),height:340}),C);

  Plotly.react('p_eq',[
    {x,y:d.series.eq_gross,type:'scattergl',mode:'lines',name:'gross (no fees)',
     line:{color:'#dde3ee',width:1.4}},
    {x,y:d.series.eq_net,type:'scattergl',mode:'lines',name:`net (${d.fee_mode})`,
     line:{color:'#d29922',width:1.4}}],
    L('mark-to-market PnL (₹ per unit size)',{shapes:shapes(0.10),height:260}),C);

  Plotly.react('p_pos',[
    {x,y:d.series.position.map(v=>v>=0?v:null),type:'scattergl',mode:'lines',
     name:'long',line:{color:'#3fb950',width:1,shape:'hv'},fill:'tozeroy'},
    {x,y:d.series.position.map(v=>v<0?v:null),type:'scattergl',mode:'lines',
     name:'short',line:{color:'#f85149',width:1,shape:'hv'},fill:'tozeroy'}],
    L('signed position (conviction-sized, max 2.0)',{height:180}),C);

  Plotly.react('p_imb',[
    {x,y:d.series.imb,type:'scattergl',mode:'lines',name:'imbalance',
     line:{color:'#bc8cff',width:0.8}},
    {x,y:d.series.drift.map(v=>0.25*v),type:'scattergl',mode:'lines',
     name:'drift sign (scaled)',line:{color:'#d29922',width:0.8}}],
    L('top-of-book imbalance vs threshold + drift confirm',
      {height:200,yaxis:{range:[-1.05,1.05],gridcolor:'#232838'},
       shapes:[+$('thr').value,-$('thr').value].map(v=>({type:'line',xref:'paper',
        x0:0,x1:1,y0:v,y1:v,line:{color:'#8b93a7',width:1,dash:'dot'}}))}),C);

  Plotly.react('p_dd',[
    {x,y:d.series.dd,type:'scattergl',mode:'lines',name:'net drawdown',
     line:{color:'#f85149',width:1},fill:'tozeroy'}],
    L(`net equity drawdown (max ${Math.min(...d.series.dd).toFixed(2)})`,{height:170}),C);

  Plotly.react('p_hist',[
    {x:d.tr_gross,type:'histogram',name:'gross',opacity:0.65,
     marker:{color:'#58a6ff'},nbinsx:40},
    {x:d.tr_net,type:'histogram',name:'net',opacity:0.65,
     marker:{color:'#d29922'},nbinsx:40}],
    L('per-trade PnL distribution',{barmode:'overlay',height:240}),C);

  Plotly.react('p_reg',[
    {x:['DIR (trend)','MM (mean-rev)','FLAT'],y:[rp['1'],rp['-1'],rp['0']],
     type:'bar',marker:{color:['#3fb950','#f85149','#8b93a7']},
     text:[rp['1'],rp['-1'],rp['0']].map(v=>v.toFixed(1)+'%'),textposition:'outside'}],
    L('regime classification (% of snapshots)',
      {height:240,showlegend:false,yaxis:{range:[0,100],gridcolor:'#232838'}}),C);
}
init();
</script></body></html>
"""


def main():
    if not glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True):
        print("no data found -- generating default NSE universe first...")
        import nse_l2_sim
        nse_l2_sim.main()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"NSE L2 backtest dashboard -> {url}   (Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
