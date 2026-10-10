"""Reproduce original engines; independently reconcile each closed trade.
This audits historical code, not a claim of deployment readiness.
"""
from pathlib import Path
import sys,json,hashlib,zipfile,ast,argparse,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
for d in ['data_pipeline','strategies/_lib','strategies/market_making','strategies/regime_lab_toolkit']:
 sys.path.insert(0,str(ROOT/'source'/d))
import l2_strategy as slow
import l2_fast_strategy as fast
import nse_l2_sim as sim
OUT=ROOT/'verification';OUT.mkdir(exist_ok=True)
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def csvwrite(name,rows):
 if rows:
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def source_inventory():
 rows=[]
 for p in sorted((ROOT/'source').rglob('*.py')):
  ast.parse(p.read_bytes());rows.append({'path':str(p.relative_to(ROOT/'source')),'sha256':sha(p)})
 save('source_manifest.json',rows)
 with zipfile.ZipFile(ROOT/'original_upload.zip') as z:
  fidelity=[dict(path=n,exact_bytes=z.read(n)==(ROOT/'source'/n).read_bytes()) for n in z.namelist() if n.endswith('.py')]
 assert len(fidelity)==47 and all(x['exact_bytes'] for x in fidelity)
 save('zip_fidelity.json',fidelity)
def behavior():
 checks=[]
 def check(name,ok,detail):
  assert ok,name;checks.append(dict(name=name,passed=bool(ok),detail=detail))
 idx=np.searchsorted([100,200],[150],side='right')-1
 check('Future trade maps to earlier snapshot',idx[0]==0,'150 ms trade assigned to 100 ms book by converter; trade time is discarded.')
 def valid(sp):
  n=len(sp);b=np.zeros((n,1,2));a=b.copy();b[:,0,0]=100;b[:,0,1]=a[:,0,1]=1;a[:,0,0]=100+sp
  return slow.validate_book(b,a,np.full(n,100.),np.arange(n))[0]
 sp=np.ones(20);sp[5]=60
 check('Validation changes when future appended',not valid(sp)[5] and valid(np.r_[sp,np.ones(40)*100])[5],'Full-file median changes earlier valid-book decision.')
 def toy(n,signal):
  mid=100+np.arange(n)*.01;b=np.zeros((n,1,2));a=b.copy();b[:,0,0]=mid-.01;a[:,0,0]=mid+.01;b[:,0,1]=a[:,0,1]=.01
  return dict(N=n,K=1,mid=mid,bb=mid-.01,ba=mid+.01,bids=b,asks=a,imb=np.full(n,.9 if signal else 0),reg=np.ones(n),drift=np.ones(n),valid=np.ones(n,dtype=bool))
 empty=slow.backtest(toy(290,False),record=True)
 check('Empty ledger miscount',empty['trades']==1 and len(empty['exits'])==0,'Original placeholder [0] is counted as a trade; audit uses actual exits.')
 op=slow.backtest(toy(250,True),record=True)
 check('Open mark differs from closed PnL',len(op['entries'])==1 and len(op['exits'])==0 and abs(op['pnl'])>1e-8,'Original total includes open inventory; audit distinguishes realized ledger.')
 d=toy(290,True);o=slow.backtest(d,record=True)
 check('Displayed depth not enforced',abs(o['position']).max()>.01,'Order size exceeds 0.01 displayed touch size; impact/partial fills absent.')
 mid=100+np.cumsum(np.random.default_rng(7).normal(0,.1,600))
 check('Fixed-cost drift prefix stable',np.array_equal(slow.drift_signal(mid[:400],.02),slow.drift_signal(mid,.02)[:400]),'Passed for this deterministic perturbation fixture; not a universal causality proof.')
 save('behavior_checks.json',checks)
def synthetic(reference):
 summaries=[];matches=[];(ROOT/'data/synthetic').mkdir(parents=True,exist_ok=True)
 for sym,kw in sim.UNIVERSE:
  d=sim.simulate(sym,N=10000,**kw);np.savez_compressed(ROOT/'data/synthetic'/f'{sym}.npz',**d)
  if reference and sym in ['INFY','TATAMOTORS']:
   p=Path(reference)/f'orderbook_{sym.lower()}_10000snaps.npz';r=np.load(p,allow_pickle=True)
   match={k:bool(np.array_equal(r[k],d[k])) for k in r.files};assert all(match.values());matches.append(dict(symbol=sym,reference_sha256=sha(p),arrays=match))
  D=slow.setup_arrays(**{k:d[k] for k in ['bids','asks','tr_price','tr_qty','tr_side','tr_bookidx','symbol']})
  for fee in [0.,2.5]:
   o=slow.backtest(D,fee_bps=fee,record=True);rows=[]
   for en,ex in zip(o['entries'],o['exits']):
    ei,side,ep=en;xi,xside,xp,reported,entry_i=ex;q=abs(o['position'][ei]);calc=side*(xp-ep)*q
    assert ei==entry_i and side==xside and np.isclose(calc,reported,atol=1e-9)
    rows.append(dict(symbol=sym,fee_bps_per_side=fee,entry_i=int(ei),exit_i=int(xi),side=int(side),quantity=float(q),entry_fee_adjusted=float(ep),exit_fee_adjusted=float(xp),pnl=float(calc),directional_hit=bool(side*(D['mid'][xi]-D['mid'][ei])>0)))
   pnl=np.array([r['pnl'] for r in rows]);res=float(o['pnl']-pnl.sum())
   assert abs(res)<1e-7,'Unexpected open inventory'
   summary=dict(symbol=sym,synthetic=True,fee_bps_per_side=fee,trades=len(rows),directional_accuracy=float(np.mean([r['directional_hit'] for r in rows])),win_rate=float(np.mean(pnl>0)),total_pnl=float(pnl.sum()),mean_pnl=float(pnl.mean()),per_trade_mean_over_std=float(pnl.mean()/pnl.std()) if pnl.std() else 0.,open_mark_residual=res)
   summaries.append(summary);csvwrite(f'synthetic_{sym}_fee{fee:g}.csv',rows)
  print('synthetic',sym,flush=True)
 save('synthetic_summary.json',summaries);save('synthetic_provenance.json',matches)
def replay(path,name):
 d=np.load(path,allow_pickle=False);bb=d['bb'].astype(float);ba=d['ba'].astype(float);bq=d['bq'].astype(float);aq=d['aq'].astype(float);ts=d['ts'];mid=(bb+ba)/2
 imb=(bq-aq)/(bq+aq);spread=(ba-bb)/mid*1e4;gate=fast.rolling_gate(ts,imb);summaries=[]
 for label,delay in fast.DELAYS:
  mv,sz,sd,tx=fast.simulate(bb,ba,ts,imb,gate,spread,1.,delay)
  candidates=np.flatnonzero((abs(imb)>=gate)&(spread<=fast.SPREAD_CAP_BPS));k=0;rows=[]
  while k<len(candidates):
   i=int(candidates[k]);fi=i+1 if delay is None else int(np.searchsorted(ts,ts[i]+delay));ei=int(np.searchsorted(ts,ts[min(fi,len(ts)-1)]+1000))
   if fi>=len(ts) or ei>=len(ts):break
   side=1 if imb[i]>0 else -1;size=min(2*np.tanh(max(1.5*(abs(imb[i])-gate[i])/max(1-gate[i],1e-9),0)),2)
   if size>0:
    ep=ba[fi] if side==1 else bb[fi];xp=bb[ei] if side==1 else ba[ei];move=side*(xp-ep)/mid[i]*1e4
    rows.append(dict(signal_i=i,entry_i=fi,exit_i=ei,entry_ms=int(ts[fi]),exit_ms=int(ts[ei]),side=side,size=float(size),entry_price=float(ep),exit_price=float(xp),move_bps=float(move),gross_weighted_bps=float(move*size),net_0_1_weighted_bps=float((move-.2)*size),net_4_5_weighted_bps=float((move-9)*size),directional_hit=bool(side*(mid[ei]-mid[fi])>0),flat_mid=bool(mid[ei]==mid[fi])))
   k=int(np.searchsorted(candidates,ei+1))
  assert np.allclose([r['move_bps'] for r in rows],mv) and np.allclose([r['size'] for r in rows],sz)
  assert np.array_equal([r['side'] for r in rows],sd) and np.array_equal([r['exit_ms'] for r in rows],tx)
  summaries.append(dict(dataset=name,input_sha256=sha(path),updates=len(ts),hold_seconds=1,delay=label,trades=len(rows),directional_accuracy=float(np.mean([r['directional_hit'] for r in rows])) if rows else None,flat_fraction=float(np.mean([r['flat_mid'] for r in rows])) if rows else None,ledger_reconciled=True,fee_scenarios={str(f):fast.metrics(mv,sz,f) for f in [0.,.1,4.5]}))
  csvwrite(f'{name}_{label}_ledger.csv',rows)
 save(f'{name}_summary.json',summaries);print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--synthetic-reference');p.add_argument('--replay');p.add_argument('--name',default='replay');a=p.parse_args()
 if a.replay:replay(a.replay,a.name)
 else:source_inventory();behavior();synthetic(a.synthetic_reference)
