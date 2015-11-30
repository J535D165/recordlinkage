import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

from recordlinkage import datasets

class TestGenerate(unittest.TestCase):

	def test_import_sample_data(self):

		dfA = datasets.load_censusA()
		dfB = datasets.load_censusB()

		# Check if length of dataframe is large than 0
		self.assertTrue(len(dfA)>0)

		# Check if length of dataframe is large than 0
		self.assertTrue(len(dfB)>0)

		# Check if index is unique
		self.assertTrue(dfA.index.is_unique)

		# Check if index is unique
		self.assertTrue(dfB.index.is_unique)
