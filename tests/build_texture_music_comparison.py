"""Build eight aligned real-music comparisons with a fixed upstream Apollo.
Run from repo root after fetching the exact sources recorded in sources.json.
This is objective/AB preparation, not an automated perceptual quality verdict.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy
from scipy.signal import resample_poly, stft
import torch
import pyloudnorm as pyln
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from texture_softening import soften_texture

BASE = 'e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6'
SR = 44100

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--sources-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['mps','cpu','cuda'], default='mps')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    upstream = a.output_dir/'upstream_inference.py'
    upstream.write_bytes(subprocess.check_output(['git','show',BASE+':inference.py']))
    spec = importlib.util.spec_from_file_location('apollo_upstream',upstream)
    api = importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    torch.set_num_threads(4)
    device = api.select_device(a.device)
    model = api.look2hear.models.BaseModel.from_pretrain(str(a.checkpoint),sr=SR,win=20,feature_dim=256,layer=6).to(device).eval()
    sources = json.loads((a.sources_dir/'sources.json').read_text())
    for row in sources:
        row['sha256'] = digest(a.sources_dir/row['file'])
    lookup = {r['id']:r for r in sources}
    cases = [('vocal', 'voice',30,False,'female vocal/sibilants; review 2–12 s'),
             ('vocal-dense','voice',70,False,'vocal with dense pop mixture; review 0–15 s'),
             ('drums','vibe',20,False,'kit/cymbals in jazz mixture; review 0–15 s'),
             ('dense','voice',110,False,'dense synth/guitar/drum mixture; review 0–15 s'),
             ('piano','piano',15,False,'piano transients/sustains; review 0–15 s'),
             ('quiet','piano',65,False,'quiet piano candidate; review decays 0–15 s'),
             ('control-piano','piano',40,True,'no added codec degradation negative control; review wanted detail'),
             ('control-celesta','celesta',10,True,'no added codec degradation negative control; review attacks')]
    report = dict(upstream_commit=BASE,upstream_inference_sha256=digest(upstream),
                  checkpoint_sha256=digest(a.checkpoint),checkpoint_revision='c68bd80fdd9c0d93d2f4a833cb154624f660a561',
                  device=str(device),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                  soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__,platform=platform.platform(),
                  chunk_seconds=6,overlap_seconds=1,chunk_pad_seconds=1,chunk_batch_size=1,
                  sources=sources,cases=[],notes='Sources are already lossy public recordings; source-proxy is NOT a lossless clean reference. No listener verdict. Controls are no-added-degradation, not certified clean recordings.')
    meter = pyln.Meter(SR)
    rng = random.Random(441044)
    for cid,sid,start,control,focus in cases:
        out = a.output_dir/cid;out.mkdir(exist_ok=True)
        raw = out/'raw';raw.mkdir(exist_ok=True)
        audition = out/'blind';audition.mkdir(exist_ok=True)
        source, rate = sf.read(a.sources_dir/lookup[sid]['file'],dtype='float32',always_2d=True)
        assert rate==SR and source.shape[1]==2
        context = source[(start-2)*SR:(start+17)*SR]
        assert context.shape==(19*SR,2)
        sf.write(out/'context-source.wav',context,SR,subtype='FLOAT')
        if control: degraded = context.copy()
        else:
            # libsndfile/LAME constant-bitrate encoding, fixed compression setting.
            sf.write(out/'context-degraded.mp3',context,SR,format='MP3',bitrate_mode='CONSTANT',compression_level=.8)
            degraded, rate = sf.read(out/'context-degraded.mp3',dtype='float32',always_2d=True)
            assert degraded.shape==context.shape and rate==SR
        sf.write(out/'context-input.wav',degraded,SR,subtype='FLOAT')
        x = torch.from_numpy(np.ascontiguousarray(degraded.T)).unsqueeze(0)
        then = time.perf_counter()
        with torch.inference_mode():
            prediction = api.run_model(model,x,device,chunk_samples=6*SR,overlap_samples=SR,chunk_pad_samples=SR,chunk_batch_size=1).squeeze(0).numpy()
        elapsed = time.perf_counter()-then
        sf.write(out/'context-apollo.wav',prediction.T,SR,subtype='FLOAT')
        variants = {'source-proxy':context[2*SR:17*SR].T,'degraded':degraded[2*SR:17*SR].T,
                    'apollo-0db':prediction[:,2*SR:17*SR]}
        stats = {}
        for cap in (1,2,3):
            then=time.perf_counter();soft,st=soften_texture(prediction,max_reduction_db=cap)
            stats[str(cap)]={**st,'seconds':time.perf_counter()-then}
            variants[f'apollo-{cap}db']=soft[:,2*SR:17*SR]
        metrics={}
        for name,v in variants.items():
            sf.write(raw/(name+'.wav'),v.T,SR,subtype='FLOAT')
            ref=variants['source-proxy'].astype(np.float64);z=v.astype(np.float64)
            gain=float((ref*z).sum()/max((z*z).sum(),1e-30))
            snr=10*np.log10(max((ref*ref).sum(),1e-30)/max(((ref-gain*z)**2).sum(),1e-30))
            _,_,rz=stft(ref,fs=SR,nperseg=2048,noverlap=1024,boundary=None,padded=False)
            _,_,vz=stft(z*gain,fs=SR,nperseg=2048,noverlap=1024,boundary=None,padded=False)
            spectral=float(np.sqrt(np.mean((20*np.log10(np.maximum(abs(rz),1e-5))-20*np.log10(np.maximum(abs(vz),1e-5)))**2)))
            metrics[name]={'proxy_snr_db':float(snr),'proxy_log_spectrum_rmse_db':spectral,'integrated_lufs':float(meter.integrated_loudness(v.T))}
        matched={name:v*10**((-23-metrics[name]['integrated_lufs'])/20) for name,v in variants.items()}
        truepeak=max(float(abs(resample_poly(v,4,1,axis=-1)).max()) for v in matched.values())
        common=min(1.,10**(-1/20)/truepeak)
        names=list(matched);rng.shuffle(names);key={}
        for i,name in enumerate(names):
            letter=chr(65+i);v=matched[name]*common
            sf.write(audition/(letter+'.wav'),v.T,SR,subtype='FLOAT')
            key[letter]=name
        row=dict(id=cid,source=sid,start_seconds=start,duration=15,no_added_degradation=control,
                 focus=focus,model_seconds=elapsed,postprocessing=stats,metrics=metrics,
                 blind_key=key,blind_target_lufs=-23+20*np.log10(common),common_gain_db=20*np.log10(common))
        report['cases'].append(row)
        (a.output_dir/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
        print(cid,'model seconds',round(elapsed,2),flush=True)
    hashes={str(f.relative_to(a.output_dir)):digest(f) for f in a.output_dir.rglob('*.wav')}
    (a.output_dir/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output_dir/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')


if __name__=='__main__': main()
