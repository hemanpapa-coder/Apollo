import argparse,sys,math,hashlib,time,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--apollo-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps');a=p.parse_args()
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()=='99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76','Unexpected checkpoint'
sys.path.insert(0,str(a.apollo_root));sys.path.insert(0,str(Path(__file__).parent/'source'));sys.modules['yaml._yaml']=None
import numpy as np,torch,soundfile as sf
from scipy.signal import resample_poly
import look2hear.models
from apollo_enhanced import restore_region
x,sr=sf.read(a.input,dtype='float32',always_2d=True)
if sr!=44100:x=resample_poly(x,44100//math.gcd(sr,44100),sr//math.gcd(sr,44100),axis=0).astype(np.float32)
torch.set_num_threads(4);device=torch.device(a.device)
m=look2hear.models.BaseModel.from_pretrain(str(a.checkpoint),sr=44100,win=20,feature_dim=256,layer=6).to(device).eval()
def forward(v):
 with torch.no_grad():return m(torch.from_numpy(v).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
t=time.perf_counter();y=restore_region(x.T,0,len(x),forward)
assert y.shape==x.T.shape and np.isfinite(y).all()
a.output.parent.mkdir(parents=True,exist_ok=True);sf.write(a.output,y.T,44100,subtype='FLOAT')
print(json.dumps({'seconds':time.perf_counter()-t,'frames':len(x),'sample_rate':44100,'device':str(device),'torch':torch.__version__}))
