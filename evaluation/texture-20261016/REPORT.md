# PR44 follow-up validation — 2026-10-16

## Early validation fixes

SciPy is imported before audio/model loading only when texture softening is
enabled. A missing dependency produces an actionable RuntimeError before
checkpoint resolution, download or forward inference. Inputs shorter than
2048 frames are rejected before checkpoint resolution/model loading when
enabled. The disabled path retains short-clip behavior and does not import the
optional processor. New tests assert these boundaries and source-file retention.
Local tests and pinned Linux CPU CI each passed all 33 tests.

Successful CI: https://github.com/hemanpapa-coder/Apollo/actions/runs/38916720169
Commit tested: 9a4cd0875af86d5cc8b8334f44f1f2a8a2e3e16d
The upstream PR run is action_required (maintainer approval), not a completed
upstream CI pass. The fork uses the same workflow and contribution commit. Its log confirms
33 tests passed in 0.095 s under the pinned dependencies. A validation-only
companion PR is https://github.com/hemanpapa-coder/Apollo/pull/1; no merge is
requested there.

## Post-process memory

Fresh-process measurements, model excluded. Tracemalloc starts after input
allocation; process high-water RSS includes Python, imports, input and output.
These are different quantities and must not be compared as if identical.

| Input | Traced peak | Process peak RSS | DSP time |
|---|---:|---:|---:|
| 30 s real music | 388.74 MiB | 648.77 MiB | 0.344 s |
| 180 s repeated Apollo prediction | 2328.58 MiB | 3042.25 MiB | 2.501 s |

The 180 s input repeats a 19 s real Apollo prediction; this is a song-length
allocation probe, not a distinct three-minute listening test. Figures are
host/version dependent. Detailed JSON records NumPy/SciPy versions and process
baseline. The full-output STFT remains O(duration), even with chunked model
inference. Multi-minute material can require several GiB in addition to the
model; there is no bounded-memory post-process implementation in this PR.

Reproduce in separate processes:

```sh
python tests/measure_texture_memory.py --seconds 30 --input /path/to/music.wav --output /path/to/memory-30.json
python tests/measure_texture_memory.py --seconds 180 --input /path/to/context-apollo.wav --output /path/to/memory-180.json
```

## Real music comparison

Eight 15-second cases from four publicly shareable CC BY recordings. The
comparison includes raw WAVs and six randomized, integrated-loudness-matched
conditions per excerpt; all inference settings and the checkpoint are fixed.
Sources are already lossy public files, so the decoded source proxy is not a
lossless clean master. Metrics against it cannot establish perceptual quality.
Listening validation and maintainer evaluation remain outstanding; keep Draft.

The quiet case is a deliberate low-level stress test: -18 dB is applied to
source context before encoding and inference (about -33.50 LUFS). All other
cases use unity input gain. Two cases add no further codec degradation; these
are negative controls, not certified artifact-free recordings.

At 3 dB, the no-added-degradation piano control has a proxy waveform SNR delta
of -0.02348 dB and spectrum RMSE delta of +0.00145 dB. The celesta control has
-0.02973 dB / +0.00846 dB respectively. These are objective adverse/caution
examples, not evidence of audible harm. The other piano case also has a very
small SNR regression. All cases are retained and none is labeled a perceptual
success. A controlled listener must judge roughness versus lost wanted detail.

The raw Apollo path is taken directly from the fixed upstream inference.py;
there is no app-specific polarity ensemble, gain fusion, dry/wet blend or
phase stabilization. Only this PR's optional texture processor varies.

Public review bundle: https://github.com/hemanpapa-coder/Apollo/releases/tag/texture-evaluation-20261016

Bundle includes 96 raw/blind excerpt WAVs plus context and metadata. Source
credits and source hashes are in sources.json. Codec alignment peak lag is
recorded per excerpt; all are zero samples. Per-case blind loudness spread and
4x-oversampled peaks are in audition-checks.json. These checks do not replace
physical listening.
