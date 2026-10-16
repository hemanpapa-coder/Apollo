# Experimental texture softening

Contribution from Neuracoust, adapted from its Sound Build-up post-processing.
Apollo remains by Kai Li and Yi Luo / Look2Hear. Existing LICENSE (CC BY-SA 4.0)
applies. This does not modify model weights or claim a new restoration model.

`python inference.py --in_wav=input.wav --out_wav=output.wav --texture-reduction-db=2`

The default is 0: the existing inference path and output remain unchanged.
The optional CPU NumPy/SciPy post-process runs once after full-file or chunked
inference. Optional SciPy is imported before audio/model loading; missing SciPy
raises an actionable error before checkpoint download. Inputs shorter than
2048 frames are rejected before checkpoint download/model loading only when
the option is enabled. The disabled path retains short-clip behavior. It needs
at least 2048 output frames. It adds no model forward passes,
but holds a full-output STFT in memory, so chunked model inference no longer
implies bounded post-processing memory. SciPy is required only when enabled.

The shared channel gain targets short positive spectral excursions relative to
a roughly 40 ms temporal median. It tapers between 1.2 and 14 kHz, limits gain
reduction to the requested 0–3 dB cap, and gates broad rising-energy frames.
Thresholds are engineering heuristics, not an artifact or vocal detector.
Normal musical modulation can trigger it; attacks can still be affected.
The cap describes STFT gain, not a bound on every time-domain sample difference.
It can reduce wanted detail and is deliberately opt-in.

## Validation and limits

Run `python -m unittest discover -s tests -p 'test_texture_softening.py' -v`.
Integration tests additionally verify that the disabled path never calls the
post-process, the enabled path writes its exact result, and the source WAV is
unchanged. Tests check exact zero-strength bypass, stationary tone preservation, common
stereo scaling including polarity, silence, exact odd output length, invalid
input, and a positive modulation case where processing actually engages.

Run `python tests/benchmark_texture_softening.py --output-dir=texture-comparison`
to generate deterministic synthetic before/after FLOAT WAVs, equal-RMS audition
copies with common peak headroom, SHA-256 provenance and timing JSON. No Apollo
checkpoint, private song, or external download is needed. This tests the
post-process only, not an improvement in Apollo restoration quality.

Prior local music experiments on a larger combined inference chain found local
maximum reductions of 1.38–1.80 dB and overall RMS reduction below 0.0011 dB.
High-band reference error increased 0.012–0.065 dB and waveform SNR was mixed.
Those private fixtures are not included, and those observations do not isolate
this proposal against current upstream Apollo. There is no controlled listening
study or evidence of universal vocal/piano improvement. Wider upstream-model,
music, transient, backend and listening evaluation remains open. This is a draft
for maintainer discussion, not a recommendation to enable by default.

Local synthetic benchmark (macOS 26.5.1 arm64, Python 3.9.6, NumPy 2.0.2,
SciPy 1.13.1): 4 s stereo took 0.0301 / 0.0250 / 0.0250 s for the
post-process alone; max STFT gain reduction 1.9105 dB. Timing is hardware and
fixture dependent. Equal-RMS audition copies matched RMS within 0.000001 dB.
