"""Experimental Apollo-only gain-conditioned fusion; not used by Simple.

The alternate prediction must already be divided by its inference input gain.
This preserves the baseline low band and takes the alternate high band. It is
not a noise detector or a guarantee of perceptual restoration quality.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def stabilize_vocal_band(approved, baseline, aligned_predictions, sample_rate=44100,
                         cutoff_hz=1800):
    """Experimental robust low-band correction from aligned Apollo passes.

    No original/dry signal. Keep the approved high-band treatment; reject a
    single errant grid prediction through a pointwise median before lowpassing
    its difference from the baseline. Not an automatic vocal detector.
    """
    predictions = np.asarray(aligned_predictions)
    if (approved.ndim != 2 or baseline.shape != approved.shape
            or predictions.ndim != 3 or predictions.shape[0] not in (4, 8)
            or predictions.shape[1:] != approved.shape
            or approved.shape[-1] < 64 or sample_rate <= 4000
            or not np.isfinite(sample_rate)
            or not np.isfinite(cutoff_hz) or not 500 <= cutoff_hz <= min(3500, sample_rate*.45)
            or not all(np.isfinite(v).all() for v in (approved, baseline, predictions))):
        raise ValueError("Expected four or eight aligned finite Apollo predictions")
    consensus = np.median(predictions, axis=0)
    delta = sosfiltfilt(butter(6, cutoff_hz, fs=sample_rate, output="sos"),
                       consensus-baseline, axis=-1)
    return (approved+delta).astype(np.float32)


def fuse_preserving_phase(baseline, gain_compensated, sample_rate=44100,
                          smoothing_ms=0):
    """Experimental shared-stereo magnitude correction, no dry signal.

    Preserve baseline STFT phase and interchannel ratios, transition from
    unchanged below 3 kHz to alternate RMS magnitude above 5 kHz.
    Both inputs are gain-compensated Apollo predictions with real context.
    """
    from scipy.signal import stft, istft
    if (baseline.ndim != 2 or baseline.shape != gain_compensated.shape
            or baseline.shape[-1] < 32
            or not np.isfinite(baseline).all()
            or not np.isfinite(gain_compensated).all()
            or not np.isfinite(sample_rate) or sample_rate <= 10000
            or not np.isfinite(smoothing_ms) or not 0 <= smoothing_ms <= 40):
        raise ValueError("Invalid Apollo predictions or sample rate")
    pad = 2048
    kwargs = dict(fs=sample_rate, nperseg=2048, noverlap=1536)
    b = np.pad(baseline, ((0, 0), (pad, pad)), mode="reflect")
    c = np.pad(gain_compensated, ((0, 0), (pad, pad)), mode="reflect")
    f, _, bs = stft(b, **kwargs)
    _, _, cs = stft(c, **kwargs)
    w = .5-.5*np.cos(np.pi*np.clip((f-3000)/2000, 0, 1))
    mb = np.sqrt(np.mean(abs(bs)**2, axis=0))
    mc = np.sqrt(np.mean(abs(cs)**2, axis=0))
    gain = (1-w[:, None])+w[:, None]*mc/np.maximum(mb, 1e-12)
    if smoothing_ms:
        # Smooth only the correction envelope, never the waveform or dry/wet
        # ratio. Default zero preserves the user-approved comparison exactly.
        from scipy.ndimage import gaussian_filter1d
        sigma = smoothing_ms*.001*sample_rate/512
        gain = np.exp(gaussian_filter1d(np.log(np.maximum(gain, 1e-12)),
                                      sigma, axis=-1, mode="nearest"))
    _, output = istft(bs*gain, **kwargs)
    return output[:, pad:pad+baseline.shape[1]].astype(np.float32)


def fuse_predictions(baseline, gain_compensated, sample_rate=44100):
    if (baseline.ndim != 2 or baseline.shape != gain_compensated.shape
            or baseline.shape[-1] < 32
            or not np.isfinite(baseline).all()
            or not np.isfinite(gain_compensated).all()
            or not np.isfinite(sample_rate) or sample_rate <= 8000):
        raise ValueError("Invalid Apollo predictions or sample rate")
    highpass = butter(4, 4000, fs=sample_rate, btype="highpass", output="sos")
    return (baseline + sosfiltfilt(
        highpass, gain_compensated - baseline, axis=-1)).astype(np.float32)
