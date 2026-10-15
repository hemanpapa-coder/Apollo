"""Reproducible synthetic DSP comparison, no model or private assets."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from texture_softening import soften_texture


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sr = 44100
    t = np.arange(4*sr)/sr
    low = .4*np.sin(2*np.pi*220*t)
    high = .02*(.1+(np.mod(t, .08)<.008))*np.sin(2*np.pi*4000*t)
    x = np.asarray([low+high, .7*low-.3*high], dtype=np.float32)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        y, stats = soften_texture(x, max_reduction_db=2)
        times.append(time.perf_counter()-start)
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'scipy': scipy.__version__,
              'sample_rate': sr, 'frames': len(t), 'cap_db': 2,
              'postprocess_seconds': times, 'statistics': stats, 'files': {}}
    rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64)**2)))
    matched = [v*(rms(x)/max(rms(v), 1e-30)) for v in (x,y)]
    common = min(1., 10**(-1/20)/max(float(abs(v).max()) for v in matched))
    for name,v in [('raw-before',x),('raw-after',y),
                   ('audition-before',matched[0]*common),('audition-after',matched[1]*common)]:
        path = args.output_dir / (name+'.wav')
        sf.write(path, v.T, sr, subtype='FLOAT')
        report['files'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'rms': rms(v), 'peak': float(abs(v).max())}
    (args.output_dir/'measurements.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
