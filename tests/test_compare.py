import unittest

import recordlinkage

import numpy as np
from numpy import nan, arange
from pandas import Series, DataFrame, MultiIndex, to_datetime

# dependencies testing specific 
import pandas.util.testing as pdt
import numpy.testing as npt
from parameterized import parameterized, param

STRING_SIM_ALGORITHMS = [
    'jaro', 'q_gram', 'cosine', 'jaro_winkler', 'dameraulevenshtein',
    'levenshtein', 'lcs', 'smith_waterman'
]

NUMERIC_SIM_ALGORITHMS = [
    'step',
    'linear',
    'squared',
    'exp',
    'gauss'
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

# nosetests -v tests/test_compare.py:TestData
class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        N_A = 100
        N_B = 100

        self.A = DataFrame({
            'age': np.random.choice(AGES, N_A),
            'given_name': np.random.choice(FIRST_NAMES, N_A),
            'lastname': np.random.choice(LAST_NAMES, N_A),
            'street': np.random.choice(STREET, N_A)
        })

        self.B = DataFrame({
            'age': np.random.choice(AGES, N_B),
            'given_name': np.random.choice(FIRST_NAMES, N_B),
            'lastname': np.random.choice(LAST_NAMES, N_B),
            'street': np.random.choice(STREET, N_B)
        })

        self.A.index.name = 'index_df1'
        self.B.index.name = 'index_df2'

        self.index_AB = MultiIndex.from_arrays(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])


# tests/test_compare.py:TestCompareApi
class TestCompareApi(TestData):
    """General unittest for the compare API."""

    def test_instance_linking(self):

        comp = recordlinkage.Compare()
        comp.string('given_name', 'given_name', method='jaro')
        comp.numeric('age', 'age', method='step', offset=3, origin=2)
        comp.numeric('age', 'age', method='step', offset=0, origin=2)
        result = comp.compute(self.index_AB, self.A, self.B)

        # returns a Series
        self.assertIsInstance(result, DataFrame)

        # resulting series has a MultiIndex
        self.assertIsInstance(result.index, MultiIndex)

        # indexnames are oke
        self.assertEqual(result.index.names,
                         [self.A.index.name, self.B.index.name])

        self.assertEqual(len(result), len(self.index_AB))

    def test_instance_dedup(self):

        comp = recordlinkage.Compare()
        comp.string('given_name', 'given_name', method='jaro')
        comp.numeric('age', 'age', method='step', offset=3, origin=2)
        comp.numeric('age', 'age', method='step', offset=0, origin=2)
        result = comp.compute(self.index_AB, self.A)

        # returns a Series
        self.assertIsInstance(result, DataFrame)

        # resulting series has a MultiIndex
        self.assertIsInstance(result.index, MultiIndex)

        # indexnames are oke
        self.assertEqual(result.index.names,
                         [self.A.index.name, self.B.index.name])

        self.assertEqual(len(result), len(self.index_AB))

    def test_label_linking(self):

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'given_name',
            'given_name',
            label='my_feature_label'
        )
        result = comp.compute(self.index_AB, self.A, self.B)

        self.assertTrue("my_feature_label" in result.columns.tolist())

    def test_label_dedup(self):

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'given_name',
            'given_name',
            label='my_feature_label'
        )
        result = comp.compute(self.index_AB, self.A)

        self.assertTrue("my_feature_label" in result.columns.tolist())

    def test_incorrect_collabels_linking(self):

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            "given_name", "not_existing_label"
        )

        with self.assertRaises(KeyError):
            comp.compute(self.index_AB, self.A, self.B)

    def test_incorrect_collabels_dedup(self):

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            "given_name", "not_existing_label"
        )

        with self.assertRaises(KeyError):
            comp.compute(self.index_AB, self.A)

    def test_compare_custom_vectorized_linking(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        B = DataFrame({'col': ['abc', 'abd', 'abc', 'abc', '123']})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'col',
            'col'
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'col',
            'col',
            label='test'
        )

        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix, columns=['test'])
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_instance_type(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        B = DataFrame({'col': ['abc', 'abd', 'abc', 'abc', '123']})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        def call(s1, s2):

            # this should raise on incorrect types
            assert isinstance(s1, numpy.ndarray)
            assert isinstance(s2, numpy.ndarray)

            return np.ones(len(s1), dtype=np.int)

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'col', 'col'
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_arguments_linking(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        B = DataFrame({'col': ['abc', 'abd', 'abc', 'abc', '123']})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x : np.ones(len(s1), dtype=np.int) * x, 
            'col', 
            'col', 
            5
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x : np.ones(len(s1), dtype=np.int) * x, 
            'col', 
            'col', 
            5, 
            label='test'
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=['test'])
        pdt.assert_frame_equal(result, expected)

        # test with kwarg
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x : np.ones(len(s1), dtype=np.int) * x, 
            'col', 
            'col', 
            x=5, 
            label='test'
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=['test'])
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_dedup(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        ix = MultiIndex.from_arrays([
            [1, 2, 3, 4, 5], [2, 3, 4, 5, 1]
        ])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'col',
            'col'
        )
        result = comp.compute(ix, A)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=np.int),
            'col',
            'col',
            label='test'
        )
        result = comp.compute(ix, A)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix, columns=['test'])
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_arguments_dedup(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        ix = MultiIndex.from_arrays([
            [1, 2, 3, 4, 5], [2, 3, 4, 5, 1]
        ])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x : np.ones(len(s1), dtype=np.int) * x, 
            'col', 
            'col', 
            5
        )
        result = comp.compute(ix, A)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x : np.ones(len(s1), dtype=np.int) * x, 
            'col', 
            'col', 
            5,
            label='test'
        )        
        result = comp.compute(ix, A)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=['test'])
        pdt.assert_frame_equal(result, expected)

# tests/test_compare.py:TestCompareExact
class TestCompareExact(TestData):
    """Test the exact comparison method."""

    def test_exact_str_type(self):

        A = DataFrame({'col': ['abc', 'abc', 'abc', 'abc', 'abc']})
        B = DataFrame({'col': ['abc', 'abd', 'abc', 'abc', '123']})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        expected = DataFrame([1, 0, 1, 1, 0], index=ix)

        comp = recordlinkage.Compare()
        comp.exact('col', 'col')
        result = comp.compute(ix, A, B)

        pdt.assert_frame_equal(result, expected)

    def test_exact_num_type(self):

        A = DataFrame({'col': [42, 42, 41, 43, nan]})
        B = DataFrame({'col': [42, 42, 42, 42, 42]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        expected = DataFrame([1, 1, 0, 0, 0], index=ix)

        comp = recordlinkage.Compare()
        comp.exact('col', 'col')
        result = comp.compute(ix, A, B)

        pdt.assert_frame_equal(result, expected)

    def test_link_exact_missing(self):

        A = DataFrame({'col': [u'a', u'b', u'c', u'd', nan]})
        B = DataFrame({'col': [u'a', u'b', u'd', nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.exact('col', 'col', label='na_')
        comp.exact('col', 'col', missing_value=0, label='na_0')
        comp.exact('col', 'col', missing_value=9, label='na_9')
        comp.exact('col', 'col', missing_value=nan, label='na_na')
        comp.exact('col', 'col', missing_value='str', label='na_str')
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series([1, 1, 0, 0, 0], index=ix, name='na_')
        pdt.assert_series_equal(result['na_'], expected)

        # Missing values as 0
        expected = Series([1, 1, 0, 0, 0], index=ix, name='na_0')
        pdt.assert_series_equal(result['na_0'], expected)

        # Missing values as 9
        expected = Series([1, 1, 0, 9, 9], index=ix, name='na_9')
        pdt.assert_series_equal(result['na_9'], expected)

        # Missing values as nan
        expected = Series([1, 1, 0, nan, nan], index=ix, name='na_na')
        pdt.assert_series_equal(result['na_na'], expected)

        # Missing values as string
        expected = Series(
            [1, 1, 0, 'str', 'str'], index=ix, name='na_str')
        pdt.assert_series_equal(result['na_str'], expected)

    def test_link_exact_disagree(self):

        A = DataFrame({'col': [u'a', u'b', u'c', u'd', nan]})
        B = DataFrame({'col': [u'a', u'b', u'd', nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.exact('col', 'col', label='d_')
        comp.exact('col', 'col', disagree_value=0, label='d_0')
        comp.exact('col', 'col', disagree_value=9, label='d_9')
        comp.exact('col', 'col', disagree_value=nan, label='d_na')
        comp.exact('col', 'col', disagree_value='str', label='d_str')
        result = comp.compute(ix, A, B)

        # disagree values as default
        expected = Series([1, 1, 0, 0, 0], index=ix, name='d_')
        pdt.assert_series_equal(result['d_'], expected)

        # disagree values as 0
        expected = Series([1, 1, 0, 0, 0], index=ix, name='d_0')
        pdt.assert_series_equal(result['d_0'], expected)

        # disagree values as 9
        expected = Series([1, 1, 9, 0, 0], index=ix, name='d_9')
        pdt.assert_series_equal(result['d_9'], expected)

        # disagree values as nan
        expected = Series([1, 1, nan, 0, 0], index=ix, name='d_na')
        pdt.assert_series_equal(result['d_na'], expected)

        # disagree values as string
        expected = Series([1, 1, 'str', 0, 0], index=ix, name='d_str')
        pdt.assert_series_equal(result['d_str'], expected)


# tests/test_compare.py:TestCompareNumeric
class TestCompareNumeric(TestData):
    """Test the numeric comparison methods."""

    def test_numeric(self):

        A = DataFrame({'col': [1, 1, 1, nan, 0]})
        B = DataFrame({'col': [1, 2, 3, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric('col', 'col', 'step', offset=2)
        comp.numeric('col', 'col', method='step', offset=2)
        comp.numeric('col', 'col', 'step', 2)
        result = comp.compute(ix, A, B)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=1)
        pdt.assert_series_equal(result[1], expected)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=2)
        pdt.assert_series_equal(result[2], expected)

    def test_numeric_with_missings(self):
        """Test missing value handling."""

        A = DataFrame({'col': [1, 1, 1, nan, 0]})
        B = DataFrame({'col': [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric('col', 'col', scale=2)
        comp.numeric('col', 'col', scale=2, missing_value=0)
        comp.numeric('col', 'col', scale=2, missing_value=123.45)
        comp.numeric('col', 'col', scale=2, missing_value=nan)
        comp.numeric('col', 'col', scale=2, missing_value='str')
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series(
            [1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

        # Missing values as 0
        expected = Series(
            [1.0, 1.0, 1.0, 0.0, 0.0], index=ix, dtype=np.float64, name=1)
        pdt.assert_series_equal(result[1], expected)

        # Missing values as 123.45
        expected = Series(
            [1.0, 1.0, 1.0, 123.45, 123.45], index=ix, name=2)
        pdt.assert_series_equal(result[2], expected)

        # Missing values as nan
        expected = Series([1.0, 1.0, 1.0, nan, nan], index=ix, name=3)
        pdt.assert_series_equal(result[3], expected)

        # Missing values as string
        expected = Series(
            [1, 1, 1, 'str', 'str'], index=ix, dtype=object, name=4)
        pdt.assert_series_equal(result[4], expected)

    @parameterized.expand(NUMERIC_SIM_ALGORITHMS)
    def test_numeric_algorithms(self, alg):

        A = DataFrame({'col': [1, 1, 1, 1, 1]})
        B = DataFrame({'col': [1, 2, 3, 4, 5]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric('col', 'col', method='step', offset=1, label='step')
        comp.numeric('col', 'col', method='linear',
                     offset=1, scale=2, label='linear')
        comp.numeric('col', 'col', method='squared',
                     offset=1, scale=2, label='squared')
        comp.numeric('col', 'col', method='exp',
                     offset=1, scale=2, label='exp')
        comp.numeric('col', 'col', method='gauss',
                     offset=1, scale=2, label='gauss')
        result_df = comp.compute(ix, A, B)

        result = result_df[alg]

        # All values between 0 and 1.
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0).all())

        if alg is not 'step':

            # sim(scale) = 0.5
            expected_bool = Series(
                [False, False, False, True, False], index=ix, name=alg)
            pdt.assert_series_equal(result == 0.5, expected_bool)

            # sim(offset) = 1
            expected_bool = Series(
                [True, True, False, False, False], index=ix, name=alg)
            pdt.assert_series_equal(result == 1.0, expected_bool)

            # sim(scale) larger than 0.5
            expected_bool = Series(
                [False, False, True, False, False], index=ix, name=alg)
            pdt.assert_series_equal(
                (result > 0.5) & (result < 1.0), expected_bool)

            # sim(scale) smaller than 0.5
            expected_bool = Series(
                [False, False, False, False, True], index=ix, name=alg)
            pdt.assert_series_equal(
                (result < 0.5) & (result >= 0.0), expected_bool)


    @parameterized.expand(NUMERIC_SIM_ALGORITHMS)
    def test_numeric_algorithms_errors(self, alg):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # scale negative
        if alg != "step":
            with self.assertRaises(ValueError):
                comp = recordlinkage.Compare()
                comp.numeric('age', 'age', method=alg, offset=2, scale=-2)
                comp.compute(self.index_AB, self.A, self.B)

            # offset negative
            with self.assertRaises(ValueError):
                comp = recordlinkage.Compare()
                comp.numeric('age', 'age', method=alg, offset=-2, scale=-2)
                comp.compute(self.index_AB, self.A, self.B)

    def test_numeric_does_not_exist(self):
        """
        Raise error is the algorithm doesn't exist.
        """

        A = DataFrame({'col': [1, 1, 1, nan, 0]})
        B = DataFrame({'col': [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()

        with self.assertRaises(ValueError):
            comp.numeric('col', 'col', method='unknown_algorithm')
            comp.compute(ix, A, B)


# tests/test_compare.py:TestCompareDates
class TestCompareDates(TestData):
    """Test the exact comparison method."""

    def test_dates(self):

        A = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            nan,
                            '2004/11/23',
                            '2010/01/10',
                            '2010/10/30']
                       )})
        B = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            '2010/12/31',
                            '2005/11/23',
                            '2010/10/01',
                            '2010/9/30']
                       )})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date('col', 'col')
        result = comp.compute(ix, A, B)[0]

        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name=0)

        pdt.assert_series_equal(result, expected)

    def test_date_incorrect_dtype(self):

        A = DataFrame({'col': ['2005/11/23',
                               nan,
                               '2004/11/23',
                               '2010/01/10',
                               '2010/10/30']})
        B = DataFrame({'col': ['2005/11/23',
                               '2010/12/31',
                               '2005/11/23',
                               '2010/10/01',
                               '2010/9/30']})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        A['col1'] = to_datetime(A['col'])
        B['col1'] = to_datetime(B['col'])

        comp = recordlinkage.Compare()
        comp.date('col', 'col1')
        self.assertRaises(ValueError, comp.compute, ix, A, B)

        comp = recordlinkage.Compare()
        comp.date('col1', 'col')
        self.assertRaises(ValueError, comp.compute, ix, A, B)

    def test_dates_with_missings(self):

        A = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            nan,
                            '2004/11/23',
                            '2010/01/10',
                            '2010/10/30']
                       )})
        B = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            '2010/12/31',
                            '2005/11/23',
                            '2010/10/01',
                            '2010/9/30']
                       )})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date('col', 'col', label='m_')
        comp.date('col', 'col', missing_value=0, label='m_0')
        comp.date('col', 'col', missing_value=123.45, label='m_float')
        comp.date('col', 'col', missing_value=nan, label='m_na')
        comp.date('col', 'col', missing_value='str', label='m_str')
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name='m_')
        pdt.assert_series_equal(result['m_'], expected)

        # Missing values as 0
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name='m_0')
        pdt.assert_series_equal(result['m_0'], expected)

        # Missing values as 123.45
        expected = Series(
            [1, 123.45, 0, 0.5, 0.5], index=ix, name='m_float')
        pdt.assert_series_equal(result['m_float'], expected)

        # Missing values as nan
        expected = Series([1, nan, 0, 0.5, 0.5], index=ix, name='m_na')
        pdt.assert_series_equal(result['m_na'], expected)

        # Missing values as string
        expected = Series(
            [1, 'str', 0, 0.5, 0.5], index=ix, dtype=object, name='m_str')
        pdt.assert_series_equal(result['m_str'], expected)

    def test_dates_with_swap(self):

        months_to_swap = [
            (9, 10, 123.45),
            (10, 9, 123.45),
            (1, 2, 123.45),
            (2, 1, 123.45)
        ]

        A = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            nan,
                            '2004/11/23',
                            '2010/01/10',
                            '2010/10/30']
                       )})
        B = DataFrame({'col':
                       to_datetime(
                           ['2005/11/23',
                            '2010/12/31',
                            '2005/11/23',
                            '2010/10/01',
                            '2010/9/30']
                       )})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date('col', 'col', label='s_')
        comp.date('col', 'col', swap_month_day=0,
                  swap_months='default', label='s_1')
        comp.date('col', 'col', swap_month_day=123.45,
                  swap_months='default', label='s_2')
        comp.date('col', 'col', swap_month_day=123.45,
                  swap_months=months_to_swap, label='s_3')
        comp.date('col', 'col', swap_month_day=nan,
                  swap_months='default', missing_value=nan, label='s_4')
        comp.date('col', 'col', swap_month_day='str', label='s_5')
        result = comp.compute(ix, A, B)

        # swap_month_day as default
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name='s_')
        pdt.assert_series_equal(result['s_'], expected)

        # swap_month_day and swap_months as 0
        expected = Series([1, 0, 0, 0, 0.5], index=ix, name='s_1')
        pdt.assert_series_equal(result['s_1'], expected)

        # swap_month_day 123.45 (float)
        expected = Series([1, 0, 0, 123.45, 0.5], index=ix, name='s_2')
        pdt.assert_series_equal(result['s_2'], expected)

        # swap_month_day and swap_months 123.45 (float)
        expected = Series(
            [1, 0, 0, 123.45, 123.45], index=ix, name='s_3')
        pdt.assert_series_equal(result['s_3'], expected)

        # swap_month_day and swap_months as nan
        expected = Series([1, nan, 0, nan, 0.5], index=ix, name='s_4')
        pdt.assert_series_equal(result['s_4'], expected)

        # swap_month_day as string
        expected = Series(
            [1, 0, 0, 'str', 0.5], index=ix, dtype=object, name='s_5')
        pdt.assert_series_equal(result['s_5'], expected)


# tests/test_compare.py:TestCompareGeo
class TestCompareGeo(TestData):
    """Test the geo comparison method."""

    def test_geo(self):

        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame({
            'lat': [52.0842455, 52.3747388, 51.9280573],
            'lng': [5.0124516, 4.7585305, 4.4203581]
        })
        B = DataFrame({
            'lat': [52.3747388, 51.9280573, 52.0842455],
            'lng': [4.7585305, 4.4203581, 5.0124516]
        })
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.geo('lat', 'lng', 'lat', 'lng',
                 method='step', offset=50)  # 50 km range
        result = comp.compute(ix, A, B)

        # Missing values as default [36.639460, 54.765854, 44.092472]
        expected = Series([1.0, 0.0, 1.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

    def test_geo_batch(self):

        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame({
            'lat': [52.0842455, 52.3747388, 51.9280573],
            'lng': [5.0124516, 4.7585305, 4.4203581]
        })
        B = DataFrame({
            'lat': [52.3747388, 51.9280573, 52.0842455],
            'lng': [4.7585305, 4.4203581, 5.0124516]
        })
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.geo('lat', 'lng', 'lat', 'lng',
                 method='step', offset=1, label='step')
        comp.geo('lat', 'lng', 'lat', 'lng', method='linear',
                 offset=1, scale=2, label='linear')
        comp.geo('lat', 'lng', 'lat', 'lng', method='squared',
                 offset=1, scale=2, label='squared')
        comp.geo('lat', 'lng', 'lat', 'lng', method='exp',
                 offset=1, scale=2, label='exp')
        comp.geo('lat', 'lng', 'lat', 'lng', method='gauss',
                 offset=1, scale=2, label='gauss')
        result_df = comp.compute(ix, A, B)

        print (result_df)

        for alg in ['step', 'linear', 'squared', 'exp', 'gauss']:

            result = result_df[alg]

            # All values between 0 and 1.
            self.assertTrue((result >= 0.0).all())
            self.assertTrue((result <= 1.0).all())

    def test_geo_does_not_exist(self):

        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame({
            'lat': [52.0842455, 52.3747388, 51.9280573],
            'lng': [5.0124516, 4.7585305, 4.4203581]
        })
        B = DataFrame({
            'lat': [52.3747388, 51.9280573, 52.0842455],
            'lng': [4.7585305, 4.4203581, 5.0124516]
        })
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()

        self.assertRaises(ValueError, comp.geo, 'lat', 'lng', 'lat', 'lng', method='unknown')


# tests/test_compare.py:TestCompareStrings
class TestCompareStrings(TestData):
    """Test the exact comparison method."""

    def test_fuzzy(self):

        A = DataFrame(
            {'col': [u'str_abc', u'str_abc', u'str_abc', nan, u'hsdkf']})
        B = DataFrame({'col': [u'str_abc', u'str_abd', u'jaskdfsd', nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string('col', 'col', method='jaro', missing_value=0)
        comp.string('col', 'col', method='q_gram', missing_value=0)
        comp.string('col', 'col', method='cosine', missing_value=0)
        comp.string('col', 'col', method='jaro_winkler', missing_value=0)
        comp.string('col', 'col', method='dameraulevenshtein', missing_value=0)
        comp.string('col', 'col', method='levenshtein', missing_value=0)
        result = comp.compute(ix, A, B)

        print (result)

        self.assertFalse(result.isnull().all(1).all(0))
        self.assertTrue((result[result.notnull()] >= 0).all(1).all(0))
        self.assertTrue((result[result.notnull()] <= 1).all(1).all(0))

    @parameterized.expand(STRING_SIM_ALGORITHMS)
    def test_incorrect_input(self, alg):

        A = DataFrame({'col': [1, 1, 1, nan, 0]})
        B = DataFrame({'col': [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        with self.assertRaises(Exception):
            comp = recordlinkage.Compare()
            comp.string('col', 'col', method=alg)
            comp.compute(ix, A, B)

    @parameterized.expand(STRING_SIM_ALGORITHMS)
    def test_string_algorithms_nan(self, alg):

        A = DataFrame({'col': [u"nan", nan, nan, nan, nan]})
        B = DataFrame({'col': [u"nan", nan, nan, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string('col', 'col', method=alg)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, 0.0, 0.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

        comp = recordlinkage.Compare()
        comp.string('col', 'col', method=alg, missing_value=nan)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, nan, nan, nan, nan], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

        comp = recordlinkage.Compare()
        comp.string('col', 'col', method=alg, missing_value=9.0)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, 9.0, 9.0, 9.0, 9.0], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

    @parameterized.expand(STRING_SIM_ALGORITHMS)
    def test_string_algorithms(self, alg):

        A = DataFrame(
            {'col': [u'str_abc', u'str_abc', u'str_abc', nan, u'hsdkf']})
        B = DataFrame({'col': [u'str_abc', u'str_abd', u'jaskdfsd', nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string('col', 'col', method=alg, missing_value=0)
        result = comp.compute(ix, A, B)[0]

        self.assertFalse(result.isnull().any())

        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

        self.assertTrue((result > 0).any())
        self.assertTrue((result < 1).any())

    def test_fuzzy_does_not_exist(self):

        A = DataFrame({'col': [1, 1, 1, nan, 0]})
        B = DataFrame({'col': [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()

        self.assertRaises(ValueError, comp.string, 'col', 'col', method='unknown_algorithm')
