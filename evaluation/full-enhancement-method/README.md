# Neuracoust full Apollo enhancement method

This companion reproduces the application processing chain; PR #44 implements only the final optional texture-softening stage. The full application method is supplied for inspection and evaluation, not represented as already integrated into upstream inference.py.

## Signal path

1. Divide the selected audio into 10-second windows with 1-second output overlap. Real neighboring context is used; file edges use reflection padding.
2. Infer positive and negative polarity with 1-second context and combine (F(x)-F(-x))/2.
3. Repeat at +6 dB input gain, then compensate that gain. Fuse high-frequency magnitudes with the baseline phase and shared stereo gain, transitioning at 3–5 kHz and smoothing the gain envelope over 12 ms.
4. Infer eight shifted contexts (0, 110, 220, 330, 55, 165, 275, 385 samples), each with both polarities and 2 seconds of neighboring context. Median consensus corrects the band below 3 kHz.
5. Apply input-guided phase stabilization below 3 kHz, bounded to 0.15 rad and gated around attacks. The reference guides phase; dry waveform/magnitude is not mixed in by this step.
6. Apply the final texture-softening stage (2 dB cap), then overlap-add restored windows.

This requires 20 neural forward passes per output window. It changes inference policy and pre/post-processing, not learned weights. It can alter wanted musical content and is not a validated artifact detector. The +6 dB branch is internally gain-compensated, not a simple playback-volume boost.

## Reproduce

Use an Apollo checkout containing the compatible look2hear model architecture and its dependencies, plus scipy, soundfile and numpy. The model used is the official checkpoint SHA-256 99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76 (revision c68bd80fdd9c0d93d2f4a833cb154624f660a561). Use upstream e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6 for the model checkout. Checkpoint weights are not bundled.

`python reproduce.py --apollo-root /path/to/Apollo --checkpoint /path/to/pytorch_model.bin --input /path/to/Untitled1017.wav --output enhanced.wav --device mps`

`python -m unittest discover -s tests -p 'test_*.py'`

The source snapshot is the current application DSP, preserved byte-for-byte. The wrapper calls restore_region directly with a fixed checkpoint and omits GUI, resource throttling, manual research settings and export processing. Device/thread performance can differ from the app. Runtime results are not physical listening evidence.

## Attribution and scope

Apollo model and original architecture: JusperLee/Apollo, https://github.com/JusperLee/Apollo. Additional inference/fusion/stabilization code: Neuracoust. Source distributed under CC BY-SA 4.0; see LICENSE. Audio has separate provenance and creator permission; no blanket audio license is asserted. The creator explicitly authorized public sharing of this Suno song and comparison outputs on 2026-10-17.
