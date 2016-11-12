import unittest

import pandas

from recordlinkage.datasets import load_febrl1, \
    load_febrl2, load_febrl3, load_febrl4, load_krebsregister

# nosetests tests/test_datasets.py:TestDatasets
class TestDatasets(unittest.TestCase):

    def test_datasets_existance(self):

        # Load all datasets
        load_febrl1()
        load_febrl2()
        load_febrl3()
        load_febrl4()

    def test_krebs_dataset(self):

        krebs_data, krebs_matches = load_krebsregister()
        krebs_data_block1, krebs_matches_block1 = load_krebsregister(1)
        krebs_data_block10, krebs_matches_block10 = load_krebsregister(10)

        # count the number of recordss
        self.assertEqual(type(krebs_data), pandas.DataFrame)
        self.assertEqual(type(krebs_matches), pandas.MultiIndex)
        self.assertEqual(len(krebs_data), 5749132)
        self.assertEqual(len(krebs_matches), 20931)

        self.assertGreater(len(krebs_data_block1), 0)
        self.assertGreater(len(krebs_data_block10), 0)

        # load not existing block
        self.assertRaises(ValueError, load_krebsregister, 11)

        # missing values
        krebs_data_block10, krebs_matches = load_krebsregister(10, missing_values=0)
        self.assertEqual(krebs_data_block10.isnull().sum().sum(), 0)


    def test_krebs_missings(self):

        # missing values
        krebs_data_block10, krebs_matches = load_krebsregister(10, missing_values=0)
        self.assertEqual(krebs_data_block10.isnull().sum().sum(), 0)


    def test_krebs_shuffle(self):

        # missing values
        krebs_data_block10, krebs_matches = load_krebsregister(10, shuffle=False)
