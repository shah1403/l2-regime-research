"""Download the original fast-strategy grid's first BTC day and verify provider checksum."""
from pathlib import Path
import urllib.request,hashlib,json,zipfile
import numpy as np,pandas as pd
root=Path(__file__).resolve().parent;dest=root/'data/binance';dest.mkdir(parents=True,exist_ok=True)
name='BTCUSDT-bookTicker-2023-09-01.zip';url='https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/'+name
p=dest/name
if not p.exists(): urllib.request.urlretrieve(url,p)
check=urllib.request.urlopen(url+'.CHECKSUM').read().decode();(dest/(name+'.CHECKSUM')).write_text(check)
h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert h==check.split()[0]
print('checksum verified',h,flush=True)
df=pd.read_csv(p,usecols=['best_bid_price','best_ask_price','best_bid_qty','best_ask_qty','event_time'])
bb=df.best_bid_price.to_numpy();ba=df.best_ask_price.to_numpy();bq=df.best_bid_qty.to_numpy();aq=df.best_ask_qty.to_numpy();ts=np.maximum.accumulate(df.event_time.to_numpy(dtype=np.int64));ok=(ba>bb)&(bq>0)&(aq>0)
np.savez_compressed(dest/'BTCUSDT_2023-09-01.npz',bb=bb[ok],ba=ba[ok],bq=bq[ok],aq=aq[ok],ts=ts[ok])
(root/'verification/btc_provenance.json').write_text(json.dumps(dict(url=url,sha256=h,raw_rows=len(df),retained_rows=int(ok.sum()),timestamp_rule='maximum.accumulate(event_time)',filter='ask>bid and positive bid/ask quantities',scope='One historical day selected from the original fixed grid; no aggregate trade tape or real fills.'),indent=2))
print('converted',len(df),int(ok.sum()),flush=True)
