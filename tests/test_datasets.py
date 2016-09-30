import unittest

from recordlinkage import datasets


class TestDatasets(unittest.TestCase):

    def test_datasets_existance(self):

        # Load all datasets
        datasets.load_febrl1()
        datasets.load_febrl2()
        datasets.load_febrl3()
        datasets.load_febrl4()
