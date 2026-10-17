import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from apollo_phase_stability import stabilize_phase

class PhaseStabilityTests(unittest.TestCase):
 def test_no_original_amplitude_blend(self):
  t=np.arange(44101)/44100
  a=np.array([.1*np.sin(2*np.pi*440*t)]*2,dtype=np.float32)
  b=a*2
  np.testing.assert_allclose(stabilize_phase(a,b),b,atol=2e-7)
 def test_common_rotation_keeps_stereo_relation(self):
  t=np.arange(44100)/44100
  a=np.array([.1*np.sin(2*np.pi*440*t)]*2,dtype=np.float32)
  carrier=.1*np.sin(2*np.pi*440*t+.08*np.sin(2*np.pi*40*t))
  b=np.array([carrier,-.4*carrier],dtype=np.float32)
  z=stabilize_phase(a,b)
  self.assertEqual(z.shape,b.shape)
  np.testing.assert_allclose(z[1],-.4*z[0],atol=2e-7)
 def test_silent_reference_does_not_erase_output(self):
  b=np.random.default_rng(1).normal(0,.01,(2,4097)).astype(np.float32)
  np.testing.assert_allclose(stabilize_phase(b*0,b),b,atol=1e-7)
 def test_bad_alignment_rejected(self):
  with self.assertRaises(ValueError):stabilize_phase(np.zeros((2,100)),np.zeros((2,101)))

if __name__=='__main__':unittest.main()
