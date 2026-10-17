import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import apollo_enhanced as e
class EnhancedTests(unittest.TestCase):
 def test_chunk_boundary_and_file_edges(self):
  x=np.tile((.1*np.sin(2*np.pi*220*np.arange(12*44100)/44100)).astype('float32'),(2,1));progress=[]
  with patch.object(e,'fuse_preserving_phase',side_effect=lambda b,c,**k:b),patch.object(e,'stabilize_vocal_band',side_effect=lambda a,b,p,**k:a),patch.object(e,'stabilize_phase',side_effect=lambda a,b:b),patch.object(e,'soften_texture',side_effect=lambda a:(a,{})):
   y=e.restore_region(x,0,x.shape[1],lambda v:v,progress.append)
  np.testing.assert_allclose(y,x,atol=1e-7)
  self.assertEqual(progress[-1],1)
  self.assertTrue(all(a<=b for a,b in zip(progress,progress[1:])))
 def test_cancel_between_model_calls(self):
  def cancel(p):
   if p>0:raise RuntimeError('cancel')
  with self.assertRaisesRegex(RuntimeError,'cancel'):
   e.restore_region(np.zeros((2,44100),dtype='float32'),0,44100,lambda x:x,cancel)
 def test_real_neighbor_context_and_gain(self):
  x=np.random.default_rng(1).normal(0,.01,(2,5*44100)).astype('float32');calls=[]
  def forward(v):calls.append(v.copy());return v
  e.restore_region(x,2*44100,3*44100,forward)
  self.assertEqual(len(calls),20)
  np.testing.assert_array_equal(calls[0],x[:,44100:4*44100])
  np.testing.assert_allclose(calls[2],calls[0]*10**(.3))
