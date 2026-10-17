import unittest
import numpy as np
from engine.apollo_texture_softening import soften_texture
class TextureTests(unittest.TestCase):
 def test_bypass(self):
  x=np.random.default_rng(4).normal(0,.1,(2,16000)).astype('float32')
  y,d=soften_texture(x,max_reduction_db=0)
  np.testing.assert_allclose(x,y,atol=1e-7)
 def test_steady_tone_and_stereo(self):
  a=.2*np.sin(2*np.pi*3000*np.arange(44100)/44100);x=np.array([a,-.3*a],dtype='float32')
  y,d=soften_texture(x)
  np.testing.assert_allclose(y[1],-.3*y[0],atol=1e-7)
  self.assertLess(np.sqrt(np.mean((x[:,2048:-2048]-y[:,2048:-2048])**2)),1e-5)
 def test_silence(self):
  y,d=soften_texture(np.zeros((2,4410)))
  self.assertEqual(float(abs(y).max()),0)
 def test_bounded(self):
  x=np.random.default_rng(7).normal(0,.1,(2,16000));y,d=soften_texture(x)
  self.assertLessEqual(d['max_reduction_db'],2)
  self.assertEqual(y.shape,x.shape)
