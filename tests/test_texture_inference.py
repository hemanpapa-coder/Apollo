"""CLI integration without checkpoint download or real model inference."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import soundfile as sf
import torch
import inference


class IntegrationTests(unittest.TestCase):
    def test_optional_postprocessing_and_default_bypass(self):
        t = np.arange(4410)/44100
        x = np.array([.4*np.sin(2*np.pi*220*t) + .02*(.1+(np.mod(t,.08)<.008))*np.sin(2*np.pi*4000*t)], dtype='float32')
        from texture_softening import soften_texture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest, checkpoint = root/'in.wav', root/'out.wav', root/'model.pt'
            checkpoint.touch()
            sf.write(source, x.T, 44100, subtype='FLOAT')
            with mock.patch.object(inference.look2hear.models.BaseModel, 'from_pretrain', return_value=torch.nn.Identity()):
                with mock.patch('texture_softening.soften_texture', wraps=soften_texture) as process:
                    inference.run_inference(source, dest, checkpoint, 'cpu')
                    process.assert_not_called()
                    y, _ = sf.read(dest, dtype='float32', always_2d=True)
                    np.testing.assert_array_equal(y.T, x)
                    inference.run_inference(source, dest, checkpoint, 'cpu', texture_reduction_db=2)
                    process.assert_called_once()
                    y, _ = sf.read(dest, dtype='float32', always_2d=True)
                    expected, _ = soften_texture(x)
                    np.testing.assert_array_equal(y.T, expected)
            original, _ = sf.read(source, dtype='float32', always_2d=True)
            np.testing.assert_array_equal(original.T, x)

    def test_invalid_cap_rejected_before_model_load(self):
        for cap in (-1, 4, np.nan, np.inf):
            with self.assertRaises(ValueError):
                inference.run_inference('unused.wav', 'out.wav', 'unused.pt', texture_reduction_db=cap)
