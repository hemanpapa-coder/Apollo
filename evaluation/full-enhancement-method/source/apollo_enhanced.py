"""Approved comparison chain, opt-in independently of original Apollo modes."""
import time
import numpy as np
from apollo_gain_fusion import fuse_preserving_phase, stabilize_vocal_band
from apollo_phase_stability import stabilize_phase
from apollo_texture_softening import soften_texture


def restore_region(audio, start, end, forward, progress=None, sample_rate=44100):
    if sample_rate != 44100 or audio.ndim != 2 or not 0 <= start < end <= audio.shape[1]:
        raise ValueError('Expected 44.1 kHz channels-by-frames region')
    if not np.isfinite(audio).all():
        raise ValueError('Nonfinite audio')
    # Bound model calls for long selections; overlap only restored predictions.
    chunk, overlap = 10*sample_rate, sample_rate
    positions = list(range(start, end, chunk-overlap))
    if len(positions)>1 and positions[-2]+chunk>=end: positions.pop()
    out=np.zeros((audio.shape[0],end-start),dtype=np.float32)
    weights=np.zeros(end-start,dtype=np.float32)
    count=len(positions)*20; done=0
    def infer(segment):
        nonlocal done
        if progress: progress(done/count)
        y=forward(np.ascontiguousarray(segment,dtype=np.float32))
        if y.shape != segment.shape or not np.isfinite(y).all():
            raise ValueError('Invalid Apollo prediction')
        done+=1
        if progress: progress(done/count)
        return y
    def segment(lo,hi):
        v=audio[:,max(0,lo):min(audio.shape[1],hi)]
        return np.pad(v,((0,0),(max(0,-lo),max(0,hi-audio.shape[1]))),mode='reflect' if v.shape[1]>1 else 'edge')
    for i,lo in enumerate(positions):
        hi=min(lo+chunk,end); n=hi-lo
        # Baseline and +6 dB prediction: exact one-second context used in trials.
        a=segment(lo-sample_rate,hi+sample_rate)
        base=.5*(infer(a)-infer(-a))[:,sample_rate:sample_rate+n]
        gain=10**(6/20); boosted=.5*(infer(a*gain)-infer(-a*gain))[:,sample_rate:sample_rate+n]/gain
        predictions=[]
        for shift in (0,110,220,330,55,165,275,385):
            v=segment(lo-2*sample_rate-shift,hi+2*sample_rate-shift)
            y=.5*(infer(v)-infer(-v));offset=2*sample_rate+shift
            predictions.append(y[:,offset:offset+n])
        y=fuse_preserving_phase(base,boosted,smoothing_ms=12)
        y=stabilize_vocal_band(y,base,np.asarray(predictions),cutoff_hz=3000)
        y=stabilize_phase(audio[:,lo:hi],y)
        y,_=soften_texture(y)
        w=np.ones(n,dtype=np.float32)
        if i: w[:min(overlap,n)]=np.linspace(0,1,min(overlap,n))
        if hi<end: w[-overlap:]=np.linspace(1,0,overlap)
        out[:,lo-start:hi-start]+=y*w;weights[lo-start:hi-start]+=w
    if progress: progress(1.)
    return out/np.maximum(weights,1e-8)


def restore(audio,start=0,end=None,progress=None):
    import apollo_engine as ae
    import resource_control
    resource_control.check_audio(audio.shape[1],audio.shape[0])
    resource_control.configure_torch(ae.torch)
    model=ae.get_model()
    def forward(x):
        began=time.monotonic();y=ae._forward(model,x);resource_control.pause(began);return y
    return restore_region(audio,start,audio.shape[1] if end is None else end,forward,progress)
