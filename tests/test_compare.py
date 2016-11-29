import unittest
from nose_parameterized import parameterized, param

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

COMPARE_ALGORITHMS = [
    # string
    param('string', 'given_name', 'given_name', method='jaro'),
    param('string', 'given_name', 'given_name', method='jarowinkler'),
    param('string', 'given_name', 'given_name', method='levenshtein'),
    param('string', 'given_name', 'given_name', method='damerau_levenshtein'),
    param('string', 'given_name', 'given_name', method='qgram'),
    param('string', 'given_name', 'given_name', method='cosine'),

    # numeric
    param('numeric', 'age', 'age', method='step', offset=3, origin=2),
    param('numeric', 'age', 'age', method='linear', offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='exp', offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='gauss', offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='squared', offset=3, scale=3, origin=2),

    # exact
    param('exact', 'given_name', 'given_name')

]


# Run all tests in this file with:
# nosetests tests/test_compare.py

# nosetests -v tests/test_compare.py:TestCompare
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

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_instance_linking(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertIsInstance(result, pandas.Series)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_instance_dedup(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertIsInstance(result, pandas.Series)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_len_result_linking(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(len(result), len(self.index_AB))

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_len_result_dedup(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(len(result), len(self.index_AB))

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_name_default_linking(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(result.name, None)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_name_default_dedup(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(result.name, None)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_name_custom_linking(self, method_to_call, *args, **kwargs):

        kwargs["name"] = "given_name_comp"

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(result.name, "given_name_comp")

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_name_custom_dedup(self, method_to_call, *args, **kwargs):

        kwargs["name"] = "given_name_comp"

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(result.name, "given_name_comp")
