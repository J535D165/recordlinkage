import unittest

import pandas.util.testing as pdt
import recordlinkage
import numpy as np
import pandas as pd

from recordlinkage.sampledata import personaldata1000A, personaldata1000B

class TestGenerate(unittest.TestCase):

    def test_import_sample_data(self):

        # Check if index is unique
        self.assertTrue(personaldata1000A.index.is_unique)

        # Check if index is unique
        self.assertTrue(personaldata1000B.index.is_unique)
