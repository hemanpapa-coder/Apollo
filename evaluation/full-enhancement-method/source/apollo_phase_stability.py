"""Experimental phase-only correction; not enabled in Simple.
Reference guides timing only: no dry samples or magnitude are mixed into output.
A shared stereo rotation up to 0.15 rad below 3 kHz reduces rapid relative phase
changes. This is not a click detector and perceptual efficacy is unverified.
"""
import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import gaussian_filter1d

def stabilize_phase(reference, processed, sample_rate=44100):
 a=np.asarray(reference);b=np.asarray(processed);sr=sample_rate
 if a.ndim!=2 or a.shape!=b.shape or a.shape[-1]<64 or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(sr) or sr<=6000:
  raise ValueError("Expected aligned finite stereo/channel arrays")
 pad=2048;kw=dict(fs=sr,nperseg=2048,noverlap=1536)
 f,t,A=stft(np.pad(a,((0,0),(pad,pad)),mode='reflect'),**kw);_,_,B=stft(np.pad(b,((0,0),(pad,pad)),mode='reflect'),**kw)
 cross=np.sum(B*np.conj(A),axis=0);weight=abs(cross)
 sm=gaussian_filter1d(cross.real,1.3,axis=-1)+1j*gaussian_filter1d(cross.imag,1.3,axis=-1)
 turn=np.angle(sm*np.conj(cross))
 w=.5+.5*np.cos(np.pi*np.clip((f-1800)/1200,0,1))
 # Only reliable input-supported bins; do not use reference magnitude in output.
 support=np.sum(abs(A)**2,axis=0)>np.maximum(np.sum(abs(B)**2,axis=0)*.04,1e-12)
 # Protect strong rising attacks across all channels.
 energy=np.sum(abs(A)**2,axis=(0,1));attack=np.r_[False,energy[1:]>1.5*np.maximum(energy[:-1],1e-15)]
 gate=1-gaussian_filter1d(attack.astype(float),1,mode='nearest')/max(gaussian_filter1d(np.array([0.,0,1,0,0]),1).max(),1e-9);gate=np.clip(gate,0,1)
 change=np.clip(turn,-.15,.15)*w[:,None]*support*gate[None,:]
 _,z=istft(B*np.exp(1j*change),**kw);z=z[:,pad:pad+b.shape[1]]
 return z.astype(np.float32)
