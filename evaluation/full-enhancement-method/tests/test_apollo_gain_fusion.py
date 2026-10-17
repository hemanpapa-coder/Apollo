import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from apollo_gain_fusion import fuse_predictions, fuse_preserving_phase, stabilize_vocal_band


class GainFusionTests(unittest.TestCase):
    def test_eight_pass_consensus_rejects_two_outliers(self):
        t = np.arange(44100)/44100
        base = np.array([.1*np.sin(2*np.pi*300*t)]*2, dtype=np.float32)
        approved = base+.03*np.sin(2*np.pi*8000*t)
        grids = np.stack([base.copy() for _ in range(8)])
        grids[1, :, 10000:20000] += .8
        grids[4, :, 10000:20000] -= .6
        np.testing.assert_allclose(stabilize_vocal_band(approved, base, grids, cutoff_hz=3000),
                                   approved, atol=1e-7)

    def test_extended_midband_keeps_upper_tone_and_rejects_bad_cutoff(self):
        t = np.arange(44100)/44100
        mid = .1*np.sin(2*np.pi*2300*t)
        high = .03*np.sin(2*np.pi*8000*t)
        base = np.array([mid, mid], dtype=np.float32)
        approved = base+high
        grids = np.stack([base*.9 for _ in range(4)])
        z = stabilize_vocal_band(approved, base, grids, cutoff_hz=3000)
        # The 8 kHz component is retained, while coherent 2.3 kHz correction
        # is stronger than the previous 1.8 kHz setting.
        section = slice(4410, -4410)
        amplitude = 2*np.mean(z[0,section]*np.sin(2*np.pi*8000*t[section]))
        self.assertAlmostEqual(amplitude, .03, places=5)
        with self.assertRaises(ValueError):
            stabilize_vocal_band(approved, base, grids, cutoff_hz=10000)

    def test_single_bad_grid_does_not_pollute_approved_output(self):
        t = np.arange(44100)/44100
        base = np.array([.1*np.sin(2*np.pi*300*t)]*2, dtype=np.float32)
        approved = base + .03*np.sin(2*np.pi*8000*t)
        grids = np.stack([base.copy() for _ in range(4)])
        grids[1, :, 10000:20000] += .8
        np.testing.assert_allclose(stabilize_vocal_band(approved, base, grids),
                                   approved, atol=1e-7)

    def test_consensus_low_band_correction_keeps_high_detail(self):
        t = np.arange(44100)/44100
        base = np.array([.1*np.sin(2*np.pi*300*t)]*2, dtype=np.float32)
        high = .03*np.sin(2*np.pi*8000*t)
        approved = base+high
        grids = np.stack([base*.8 for _ in range(4)])
        result = stabilize_vocal_band(approved, base, grids)
        np.testing.assert_allclose(result[:,3000:-3000], (base*.8+high)[:,3000:-3000], atol=1e-6)

    def test_phase_fusion_keeps_stereo_relation_and_length(self):
        t = np.arange(44101)/44100
        tone = np.sin(2*np.pi*8000*t)
        base = np.array([tone, -.3*tone], dtype=np.float32)
        alternate = np.array([.5*tone, .5*tone], dtype=np.float32)
        result = fuse_preserving_phase(base, alternate)
        self.assertEqual(result.shape, base.shape)
        np.testing.assert_allclose(result[1], -.3*result[0], atol=2e-6)
        self.assertTrue(np.isfinite(result).all())

    def test_phase_fusion_silence_and_identity(self):
        x = np.random.default_rng(2).normal(size=(2, 4097)).astype(np.float32)
        np.testing.assert_allclose(fuse_preserving_phase(x, x), x, atol=1e-6)
        z = np.zeros((2, 40), dtype=np.float32)
        np.testing.assert_array_equal(fuse_preserving_phase(z, z), z)

    def test_band_routing_and_stereo(self):
        t = np.arange(44100) / 44100
        low, high = np.sin(2*np.pi*200*t), np.sin(2*np.pi*12000*t)
        base = np.array([low + high, .5*low - high], dtype=np.float32)
        alt = np.array([.1*low + .7*high, .1*low - .7*high], dtype=np.float32)
        result = fuse_predictions(base, alt)[:, 4410:-4410]
        expected = np.array([low+.7*high, .5*low-.7*high])[:, 4410:-4410]
        self.assertLess(np.max(abs(result-expected)), .001)

    def test_identity_and_validation(self):
        x = np.random.default_rng(1).normal(size=(2, 1000)).astype(np.float32)
        np.testing.assert_array_equal(fuse_predictions(x, x), x)
        with self.assertRaises(ValueError):
            fuse_predictions(x, x[:, :-1])
        with self.assertRaises(ValueError):
            fuse_predictions(x, x * np.nan)


if __name__ == "__main__":
    unittest.main()
