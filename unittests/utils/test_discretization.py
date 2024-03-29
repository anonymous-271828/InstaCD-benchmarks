import numpy as np
import pandas as pd
from instacd_benchmarks.utils.entropy_discretization import entropy_discretize, convert_to_continuous_bins
import unittest

class TestDiscretization(unittest.TestCase):

   def test_discretize_normal(self):
      # Generate synthetic data
      np.random.seed(42)
      data = pd.Series(np.random.rand(100))
      # Multi-class target based on data quartiles
      target = pd.cut(data, bins=4, labels=False)  
      
      bins = entropy_discretize(data, target, max_depth=2)
      converted_bins = convert_to_continuous_bins(bins)
      discretized = pd.Series(np.digitize(data, converted_bins), index=data.index)
      self.assertListEqual(list(discretized.values), [1, 3, 2, 2, 0, 0, 0, 3, 2, 2, 0, 3, 3, 0, 0, 0, 1, 2, 1, 1, 2, 0, 1, 1, 1, 3, 0, 2, 2, 0, 2, 0, 0, 3, 3, 3, 1, 0, 2, 1, 0, 1, 0, 3, 1, 2, 1, 2, 2, 0, 3, 3, 3, 3, 2, 3, 0, 0, 0, 1, 1, 1, 3, 1, 1, 2, 0, 3, 0, 3, 3, 0, 0, 3, 2, 2, 3, 0, 1, 0, 3, 2, 1, 0, 1, 1, 2, 2, 3, 1, 0, 2, 3, 2, 3, 1, 2, 1, 0, 0])