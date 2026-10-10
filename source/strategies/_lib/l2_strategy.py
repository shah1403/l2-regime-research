"""Regime-switching order-book strategy (user-provided, verbatim logic) plus an
instrumented backtest that records the full equity / position / trade time series
so the runner can plot PnL, mid-with-trades, positions and regime shading.

.npz format: bids,asks (N,K,2)=[price,size]; tr_price,tr_qty,
tr_side(+1 aggressor-buy/-1 sell),tr_bookidx; symbol.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

# =============================== config ===============================
DET = dict(W=100, vr_k=8, h_dir=10, vr_hi=1.10, vr_lo=0.88, ac_min=0.06, hyst=3)
WINDOWS = [20, 40, 60]; ORDERS = [1, 2]; HORIZONS = [5, 10, 20, 30]; L = 150
WARM = max(WINDOWS) + max(HORIZONS) + L
SPREAD_MULT = 50      # spread > this x median => invalid snapshot
TRADE_TOL   = 2.0     # price tolerance for "last trade near the touch"
LAG         = 1       # decide at t, fill at t+LAG (no same-snapshot leak)
# sizing constants (locked; not tuned on the OOS file)
SIZE_K      = 1.5
SIZE_MAX    = 2.0

# ========================= regime detector ============================
def variance_ratio(r, k):
    if len(r) < k + 2: return 1.0
    v1 = np.var(r, ddof=1)
    if v1 <= 0: return 1.0
    rk = np.convolve(r, np.ones(k), 'valid')
    return np.var(rk, ddof=1) / (k * v1)

def autocorr1(x):
    if len(x) < 3: return 0.0
    x = x - x.mean(); d = (x[:-1]**2).sum()
    return float((x[:-1]*x[1:]).sum() / d) if d > 0 else 0.0

def compute_regime(mid, spread, signed_flow, W=100, vr_k=8, h_dir=10,
                   vr_hi=1.10, vr_lo=0.88, ac_min=0.06, cost_mult=1.0, hyst=3):
    N=len(mid); r=np.diff(mid, prepend=mid[0]); raw=np.zeros(N,int)
    for t in range(W,N):
        seg_r=r[t-W:t]; seg_f=signed_flow[t-W:t]
        vr=variance_ratio(seg_r,vr_k); ac=autocorr1(seg_r); fp=autocorr1(seg_f)
        exp_move=seg_r.std()*np.sqrt(h_dir); cost=spread[t]*cost_mult
        trending =(vr>vr_hi) and (ac> ac_min) and (exp_move>cost) and (fp>0)
        reverting=(vr<vr_lo) and (ac<-ac_min)
        raw[t]=1 if trending else (-1 if reverting else 0)
    reg=np.zeros(N,int); cur=0; run=0
    for t in range(N):
        if raw[t]==cur: run=0
        else:
            run+=1
            if run>=hyst: cur=raw[t]; run=0
        reg[t]=cur
    return reg

# ===================== edge-selected kinematic drift ==================
def _proj(W,d,lam):
    tau=np.arange(-(W-1),1,dtype=float); w=np.exp(tau/lam)
    Vd=np.vander(tau,d+1,increasing=True)
    return np.linalg.solve(Vd.T@(Vd*w[:,None]),(Vd*w[:,None]).T)
_FUT={(O,H):np.vander(np.arange(1,H+1),O+1,increasing=True) for O in ORDERS for H in HORIZONS}

def drift_signal(mid, cost):
    N=len(mid); COEF={}
    for W in WINDOWS:
        wv=swv(mid,W)
        for O in ORDERS:
            C=wv@_proj(W,O,W/2).T; full=np.full((N,O+1),np.nan); full[W-1:]=C; COEF[(W,O)]=full
    combos=[(W,O,H) for W in WINDOWS for O in ORDERS for H in HORIZONS]
    recent=np.full((len(combos),N),-1e9); disp=np.zeros((len(combos),N))
    for i,(W,O,H) in enumerate(combos):
        C=COEF[(W,O)]; dd=np.nan_to_num((C@_FUT[(O,H)][-1])-C[:,0]); disp[i]=dd
        d=np.sign(dd); rz=np.full(N,np.nan); rz[:N-H]=mid[H:]-mid[:N-H]
        edge=np.nan_to_num(d*rz-cost); cs=np.cumsum(edge); ce=np.full(N,-1e9)
        for t in range(H+L,N): ce[t]=cs[t-H]-cs[t-H-L]
        recent[i]=ce
    sel=np.argmax(recent,axis=0)
    return np.array([np.sign(disp[sel[t],t]) for t in range(N)])

# ========================= book validation ============================
def validate_book(bids, asks, tr_price, tr_bookidx):
    N=bids.shape[0]; bb=bids[:,0,0].copy(); ba=asks[:,0,0].copy(); spread=ba-bb
    med=np.median(spread[spread>0]) if (spread>0).any() else 0.0
    order=np.argsort(tr_bookidx,kind='stable'); tps,tbs=tr_price[order],tr_bookidx[order]
    last_trade=np.full(N,np.nan); lt=np.nan; ptr=0
    for t in range(N):
        while ptr<len(tps) and tbs[ptr]<=t: lt=tps[ptr]; ptr+=1
        last_trade[t]=lt
    valid=((spread>0)&(spread<SPREAD_MULT*med)&
           (bids[:,0,1]>0)&(asks[:,0,1]>0)&(~np.isnan(last_trade))&
           (bb-TRADE_TOL<=last_trade)&(last_trade<=ba+TRADE_TOL))
    last_ok=-1                                   # forward-fill touch for signal continuity
    for t in range(N):
        if valid[t]: last_ok=t
        elif last_ok>=0: bb[t]=bb[last_ok]; ba[t]=ba[last_ok]
    return valid, bb, ba

# ============================== setup =================================
def setup(path):
    raw=np.load(path,allow_pickle=True)
    sym=str(raw['symbol']) if 'symbol' in raw else '?'
    return setup_arrays(raw['bids'],raw['asks'],raw['tr_price'],raw['tr_qty'],
                        raw['tr_side'],raw['tr_bookidx'],sym)

def setup_arrays(bids, asks, tr_price, tr_qty, tr_side, tr_bookidx, symbol='?'):
    """Full signal pipeline on in-memory arrays (lets a dashboard backtest an
    arbitrary snapshot slice with signals recomputed strictly inside the slice)."""
    N,K,_=bids.shape
    tp=tr_price.astype(float); tq=tr_qty.astype(float)
    tsd=tr_side.astype(int);  tbi=tr_bookidx.astype(int)
    valid,bb,ba=validate_book(bids,asks,tp,tbi)
    mid=(bb+ba)/2; spread=ba-bb
    bsz=bids[:,0,1].copy(); asz=asks[:,0,1].copy()          # imbalance from valid snaps (ffill)
    for t in range(1,N):
        if not valid[t]: bsz[t]=bsz[t-1]; asz[t]=asz[t-1]
    imb=(bsz-asz)/(bsz+asz+1e-9)
    tnf=np.zeros(N)
    for j in range(len(tp)):
        t=int(tbi[j])
        if 0<=t<N: tnf[t]+=tsd[j]*tq[j]
    reg=compute_regime(mid,spread,tnf,**DET)
    drift=drift_signal(mid, cost=np.median(spread[spread>0]) if (spread>0).any() else 0.0)
    return dict(N=N,K=K,mid=mid,bb=bb,ba=ba,bids=bids,asks=asks,spread=spread,
                imb=imb,reg=reg,drift=drift,valid=valid,symbol=symbol,tnf=tnf)

# ============================== fills =================================
def _cross_buy(D,t):
    for k in range(D['K']):
        if D['asks'][t,k,1]>0: return D['asks'][t,k,0]
    return D['ba'][t]
def _cross_sell(D,t):
    for k in range(D['K']):
        if D['bids'][t,k,1]>0: return D['bids'][t,k,0]
    return D['bb'][t]

# ======================= conviction sizing ============================
def size_of(imb_val, thr, sizing):
    """0 at |imb|=thr, saturating toward SIZE_MAX as |imb|->1."""
    if sizing=='binary': return 1.0
    z=SIZE_K*(abs(imb_val)-thr)/max(1-thr,1e-9); z=max(z,0.0)
    if sizing=='arctan': return float(np.clip(SIZE_MAX*(2/np.pi)*np.arctan(z),0,SIZE_MAX))
    return float(np.clip(SIZE_MAX*np.tanh(z),0,SIZE_MAX))     # tanh (default)

# ============================ strategy ================================
def backtest(D, use_drift_confirm=True, thr=0.4, hold=15, fee_bps=0.0, sizing='tanh',
             record=False, use_regime=True):
    """Same decision/fill logic as the original; with record=True also returns the
    mark-to-market equity curve, signed position series, and per-trade event marks.
    `thr` may be a scalar or a per-snapshot array (walk-forward dynamic threshold).
    `use_regime=False` ablates the DIR gate (entries allowed in any regime)."""
    N=D['N']; imb=D['imb']; reg=D['reg']; drift=D['drift']; valid=D['valid']; mid=D['mid']
    thr_a=np.broadcast_to(np.asarray(thr,float),(N,))
    f=fee_bps/1e4; cash=0.; q=0.; pos=0; entry=0.; held=0; psz=0.; tr=[]
    fills=[]   # (idx, dq, cash_delta) applied at fill index te
    entries=[]; exits=[]                       # (te, side, px) / (te, side, px, pnl)
    entry_te=0
    for t in range(WARM, N-LAG):
        te=t+LAG; th=thr_a[t]
        isig=np.sign(imb[t]) if abs(imb[t])>th else 0
        s=(isig if (isig!=0 and isig==drift[t]) else 0) if use_drift_confirm else isig
        if pos==0:
            if (reg[t]==1 or not use_regime) and s!=0 and valid[te]:
                m=size_of(imb[t],th,sizing)
                if m<=0: continue
                pos=int(s); psz=m; held=0
                if pos>0: px=_cross_buy(D,te)*(1+f);  q+=m; cash-=px*m; fills.append((te, m,-px*m))
                else:     px=_cross_sell(D,te)*(1-f); q-=m; cash+=px*m; fills.append((te,-m, px*m))
                entry=px; entry_te=te; entries.append((te,pos,px))
        else:
            held+=1
            if (held>=hold or (reg[t]!=1 and use_regime)) and valid[te]:
                if pos>0:
                    px=_cross_sell(D,te)*(1-f); cash+=px*abs(q); fills.append((te,-abs(q), px*abs(q)))
                    tr.append((px-entry)*psz);  exits.append((te,pos,px,(px-entry)*psz,entry_te))
                else:
                    px=_cross_buy(D,te)*(1+f);  cash-=px*abs(q); fills.append((te, abs(q),-px*abs(q)))
                    tr.append((entry-px)*psz);  exits.append((te,pos,px,(entry-px)*psz,entry_te))
                q=0; pos=0
    tr=np.array(tr) if tr else np.array([0.0])
    pts=tr.mean()/tr.std() if tr.std()>0 else 0.0
    out=dict(pnl=cash+q*mid[-1], per_trade_sharpe=pts, tstat=pts*np.sqrt(len(tr)),
             trades=len(tr), win_rate=100*np.mean(tr>0), edge_bps=tr.mean()/np.median(mid)*1e4,
             trade_pnls=tr)
    if record:
        q_ts=np.zeros(N); cash_ts=np.zeros(N)
        for te,dq,dc in fills:
            q_ts[te:]+=dq; cash_ts[te:]+=dc
        out['equity']=cash_ts+q_ts*mid           # mark-to-market PnL series
        out['position']=q_ts
        out['entries']=entries; out['exits']=exits
        out['fills']=fills                       # (idx, dq, cash_delta) per fill leg
    return out
