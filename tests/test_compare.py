import unittest

import pandas.util.testing as pdt
import numpy.testing as npt
import recordlinkage

import numpy as np
from numpy import nan, arange
import pandas

STRING_SIM_ALGORITHMS = [
    'jaro', 'q_gram', 'cosine', 'jaro_winkler', 'dameraulevenshtein',
    'levenshtein'
]

NUMERIC_SIM_ALGORITHMS = [
    'step', 'linear', 'squared', 'exp', 'gauss'
]

# Run all tests in this file with:
# nosetests tests/test_compare.py

# nosetests tests/test_compare.py:TestCompare
class TestCompare(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.A = pandas.DataFrame({
            'age': [20.0, 17.0, 33.0, 27.0, nan],
            'given_name': [u'Donell', nan, u'Kalie', u'Kittie', nan],
            'lastname': [u'Gerlach', u'Smit', u'Flatley', u'Schuster', nan],
            'place': [u'New York', u'Boston', u'Boston', nan, u'South Devyn']
        })

        self.A.index.name = 'index_df1'

        self.B = pandas.DataFrame({
            'age': [20, 17, 33, 20, 70],
            'given_name': [u'Donel', nan, u'Kaly', u'Kittie', u'Bob'],
            'lastname': [u'Gerleach', u'Smith', u'Flatley', nan, u'Armstrong'],
            'place': [u'New York', u'Boston', u'Boston', nan, u'Lake Gavinmouth']
        })

        self.B.index.name = 'index_df2'

        self.index_AB = pandas.MultiIndex.from_arrays(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])

    def test_instance_linking(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        result = comp.exact('given_name', 'given_name')
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, None)

        result = comp.numeric('age', 'age', offset=2, scale=2)
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, None)

        for alg in STRING_SIM_ALGORITHMS:

            # Missing values
            result = comp.string('given_name', 'given_name', method=alg)

            self.assertIsInstance(result, pandas.Series)
            self.assertEqual(result.name, None)

    def test_instance_dedup(self):

        comp = recordlinkage.Compare(self.index_AB, self.A)

        result = comp.exact('given_name', 'given_name')
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, None)

        result = comp.numeric('age', 'age', offset=2, scale=2)
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, None)

        for alg in STRING_SIM_ALGORITHMS:

            # Missing values
            result = comp.string('given_name', 'given_name', method=alg)

            self.assertIsInstance(result, pandas.Series)
            self.assertEqual(result.name, None)

    def test_name_series_linking(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        result = comp.exact('given_name', 'given_name', name="given_name_comp")
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, "given_name_comp")

        result = comp.numeric('age', 'age', offset=2, scale=2, name="given_name_comp")
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, "given_name_comp")

        for alg in STRING_SIM_ALGORITHMS:

            # Missing values
            result = comp.string('given_name', 'given_name',
                                 method=alg, name="given_name_comp")

            self.assertIsInstance(result, pandas.Series)
            self.assertEqual(result.name, "given_name_comp")

    def test_name_series_dedup(self):

        comp = recordlinkage.Compare(self.index_AB, self.A)

        result = comp.exact('given_name', 'given_name', name="given_name_comp")
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, "given_name_comp")

        result = comp.numeric('age', 'age', offset=2, scale=2, name="given_name_comp")
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, "given_name_comp")

        for alg in STRING_SIM_ALGORITHMS:

            # Missing values
            result = comp.string('given_name', 'given_name',
                                 method=alg, name="given_name_comp")

            self.assertIsInstance(result, pandas.Series)
            self.assertEqual(result.name, "given_name_comp")

    def test_batch_compare(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B, batch=True)

        comp.exact('given_name', 'given_name', name="given_name_comp")
        comp.exact('lastname', 'lastname', name="lastname_comp")
        comp.numeric('age', 'age', offset=2, scale=2, name="age_comp")
        comp.run()

        self.assertIsInstance(comp.vectors, pandas.DataFrame)
        self.assertEqual(
            comp.vectors.columns.tolist(),
            ["given_name_comp", "lastname_comp", "age_comp"])

        for v in comp.vectors.columns:
            self.assertTrue(comp.vectors[v].notnull().any())

    def test_batch_compare_error(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B, batch=True)

        self.assertRaises(Exception, comp.run)

