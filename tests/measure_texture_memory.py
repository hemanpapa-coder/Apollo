"""Measure post-process allocations and process high-water RSS in a fresh process.
RSS includes interpreter/dependencies and input; tracemalloc starts after input.
No model is loaded. Supply a song WAV with --input or a deterministic probe.
"""
import argparse
import json
import platform
import resource
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import scipy
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from texture_softening import soften_texture

p = argparse.ArgumentParser(__doc__)
p.add_argument('--seconds', type=float, required=True)
p.add_argument('--input', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sr = 44100
n = int(a.seconds*sr)
if a.input:
    source, sr = sf.read(a.input, dtype='float32', always_2d=True)
    if sr != 44100 or source.shape[1] != 2:
        raise ValueError('Expected 44.1 kHz stereo source')
    x = np.tile(source.T, (1, (n+len(source)-1)//len(source)))[:, :n].copy()
else:
    t = np.arange(n)/sr
    x = np.array([.2*np.sin(2*np.pi*220*t), .15*np.sin(2*np.pi*331*t)], dtype='float32')
    del t
unit = 1 if sys.platform == 'darwin' else 1024
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit
tracemalloc.start()
began = time.perf_counter()
y, stats = soften_texture(x)
seconds = time.perf_counter()-began
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
report = dict(duration=a.seconds, frames=n, channels=2, sample_rate=sr,
              numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
              input_file=a.input.name if a.input else None, input_repeated=bool(a.input and n>len(source)),
              seconds=seconds, traced_peak_bytes=peak, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit,
              process_peak_before_bytes=before, statistics=stats, finite=bool(np.isfinite(y).all()))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
