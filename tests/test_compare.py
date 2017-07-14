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
    param('string', 'given_name', 'given_name', method='lcs', norm='dice'),
    param('string', 'given_name', 'given_name', method='lcs', norm='jaccard'),
    param('string', 'given_name', 'given_name', method='lcs', norm='overlap'),

    # numeric
    param('numeric', 'age', 'age', method='step',
          offset=3, origin=2),
    param('numeric', 'age', 'age', method='linear',
          offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='exp',
          offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='gauss',
          offset=3, scale=3, origin=2),
    param('numeric', 'age', 'age', method='squared',
          offset=3, scale=3, origin=2),

    # exact
    param('exact', 'given_name', 'given_name'),
    param('exact', 'given_name', 'given_name', missing_value=9),
    param('exact', 'given_name', 'given_name',
          disagree_value=9, missing_value=9),
    param('exact', 'given_name', 'given_name',
          agree_value=9, missing_value=9)
]


FIRST_NAMES = [u'Ronald', u'Amy', u'Andrew', u'William', u'Frank', u'Jessica',
               u'Kevin', u'Tyler', u'Yvonne', nan]
LAST_NAMES = [u'Graham', u'Smith', u'Holt', u'Pope', u'Hernandez',
              u'Gutierrez', u'Rivera', nan, u'Crane', u'Padilla']
STREET = [u'Oliver Neck', nan, u'Melissa Way', u'Sara Dale',
          u'Keith Green', u'Olivia Terrace', u'Williams Trail',
          u'Durham Mountains', u'Anna Circle', u'Michelle Squares']
JOB = [u'Designer, multimedia', u'Designer, blown glass/stained glass',
       u'Chiropractor', u'Engineer, mining', u'Quantity surveyor',
       u'Phytotherapist', u'Teacher, English as a foreign language',
       u'Electrical engineer', u'Research officer, government', u'Economist']
AGES = [23, 40, 70, 45, 23, 57, 38, nan, 45, 46]


# Run all tests in this file with:
# nosetests tests/test_compare.py

# nosetests -v tests/test_compare.py:TestCompare
class TestCompare(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        N_A = 100
        N_B = 100

        self.A = pandas.DataFrame({
            'age': np.random.choice(AGES, N_A),
            'given_name': np.random.choice(FIRST_NAMES, N_A),
            'lastname': np.random.choice(LAST_NAMES, N_A),
            'street': np.random.choice(STREET, N_A)
        })

        self.B = pandas.DataFrame({
            'age': np.random.choice(AGES, N_B),
            'given_name': np.random.choice(FIRST_NAMES, N_B),
            'lastname': np.random.choice(LAST_NAMES, N_B),
            'street': np.random.choice(STREET, N_B)
        })

        self.A.index.name = 'index_df1'
        self.B.index.name = 'index_df2'

        self.index_AB = pandas.MultiIndex.from_arrays(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_instance_linking(self, method_to_call, *args, **kwargs):
        """result is pandas series (link)"""

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        # returns a pandas.Series
        self.assertIsInstance(result, pandas.Series)

        # resulting series has a pandas.MultiIndex
        self.assertIsInstance(result.index, pandas.MultiIndex)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_instance_dedup(self, method_to_call, *args, **kwargs):
        """result is pandas series (dedup)"""

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        # returns a pandas.Series
        self.assertIsInstance(result, pandas.Series)

        # resulting series has a pandas.MultiIndex
        self.assertIsInstance(result.index, pandas.MultiIndex)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_index_names_linking(self, method_to_call, *args, **kwargs):
        """result is pandas series (link)"""

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        # returns a pandas.Series
        self.assertEqual(result.index.names,
                         [self.A.index.name, self.B.index.name])

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_len_result_linking(self, method_to_call, *args, **kwargs):
        """result has correct number of records (link)"""

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        self.assertEqual(len(result), len(self.index_AB))

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_len_result_dedup(self, method_to_call, *args, **kwargs):
        """result has correct number of records (dedup)"""

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

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_incorrect_labels_linking(self, method_to_call, s1, s2, *args, **kwargs):

        kwargs["name"] = "given_name_comp"
        s2 = "not_existing_label"

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        with self.assertRaises(KeyError):
            getattr(comp, method_to_call)(s1, s2, *args, **kwargs)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_incorrect_labels_dedup(self, method_to_call, s1, s2, *args, **kwargs):

        kwargs["name"] = "given_name_comp"
        s2 = "not_existing_label"

        comp = recordlinkage.Compare(self.index_AB, self.A)

        with self.assertRaises(KeyError):
            getattr(comp, method_to_call)(s1, s2, *args, **kwargs)

    def test_deprecated_run(self):

        comp = recordlinkage.Compare(
            self.index_AB, self.A, self.B, batch_compare=True
        )

        with self.assertRaises(AttributeError):
            comp.run()

    def test_memory_low(self):

        c = recordlinkage.Compare(
            self.index_AB, self.A, self.B, low_memory=True
        )

        c.numeric('age', 'age', method='linear', offset=0, scale=3)
        c.exact('given_name', 'given_name')
        c.string('lastname', 'lastname', method='levenshtein')
        c.string('street', 'street', method='levenshtein')

        nrows, ncols = c.vectors.shape
        self.assertEqual(nrows, len(self.index_AB))
        self.assertEqual(ncols, 4)

    def test_memory_high(self):

        c = recordlinkage.Compare(
            self.index_AB, self.A, self.B, low_memory=False
        )

        c.numeric('age', 'age', method='linear', offset=0, scale=3)
        c.exact('given_name', 'given_name')
        c.string('lastname', 'lastname', method='levenshtein')
        c.string('street', 'street', method='levenshtein')

        # Check if all columns are there
        nrows, ncols = c.vectors.shape
        self.assertEqual(nrows, len(self.index_AB))
        self.assertEqual(ncols, 4)

        # Check is the records are there
        self.assertIsNotNone(c._df_a_indexed)
        self.assertIsNotNone(c._df_b_indexed)

        # Check memory clearing function
        c.clear_memory()
        self.assertIsNone(c._df_a_indexed)
        self.assertIsNone(c._df_b_indexed)

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_block_size_linking(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB,
                                     self.A, self.B, block_size=10)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        # returns a pandas.Series
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(len(result), len(self.index_AB))

    @parameterized.expand(COMPARE_ALGORITHMS)
    def test_block_size_dedup(self, method_to_call, *args, **kwargs):

        comp = recordlinkage.Compare(self.index_AB, self.A, block_size=10)
        result = getattr(comp, method_to_call)(*args, **kwargs)

        # returns a pandas.Series
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(len(result), len(self.index_AB))

    def test_series_argument(self):

        # The columns with data
        col_A = self.A['given_name']
        col_B = self.B['given_name']

        pairs = pandas.MultiIndex.from_arrays(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])

        c = recordlinkage.Compare(
            pairs, self.A, self.B, low_memory=True
        )

        # Check exact with column arguments
        result = c.exact(col_A, col_B)

        # check result is series
        self.assertIsInstance(result, pandas.Series)

        # resulting series has a pandas.MultiIndex
        self.assertIsInstance(result.index, pandas.MultiIndex)

        # resulting series has a pandas.MultiIndex
        self.assertEqual(len(result), len(col_A))


def ones_compare(s1, s2):

    return np.ones(len(s1))


# nosetests -v tests/test_compare.py:TestCompareLarge
class TestCompareLarge(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.n_records = 2131

        self.A = pandas.DataFrame({
            'var1': np.arange(self.n_records),
            'var2': np.arange(self.n_records),
            'var3': np.arange(self.n_records),
            'var4': np.arange(self.n_records)
        })
        self.A.index.name = 'index_df1'

        self.B = pandas.DataFrame({
            'var1': np.arange(self.n_records),
            'var2': np.arange(self.n_records),
            'var3': np.arange(self.n_records),
            'var4': np.arange(self.n_records)
        })
        self.B.index.name = 'index_df2'

        self.index_AB = pandas.MultiIndex.from_product(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])

    def test_instance_linking(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = comp.compare(ones_compare, 'var1', 'var1')

        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(len(result), self.n_records**2)
        self.assertTrue(result.index.is_unique)

    def test_instance_dedup(self):

        comp = recordlinkage.Compare(self.index_AB, self.A)
        result = comp.compare(ones_compare, 'var1', 'var1')

        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(len(result), self.n_records**2)
        self.assertTrue(result.index.is_unique)

