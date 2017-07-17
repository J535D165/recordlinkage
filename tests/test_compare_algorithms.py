import unittest

import pandas.util.testing as pdt
import recordlinkage

import numpy as np
from numpy import nan, arange
import pandas

STRING_SIM_ALGORITHMS = [
    'jaro', 'q_gram', 'cosine', 'jaro_winkler', 'dameraulevenshtein',
    'levenshtein', 'lcs', 'smith_waterman'
]

NUMERIC_SIM_ALGORITHMS = [
    'step', 'linear', 'squared', 'exp', 'gauss'
]

# nosetests tests/test_compare_algorithms.py:TestCompareAlgorithms
class TestCompareAlgorithms(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.A = pandas.DataFrame([
            [u'Donell', u'Gerlach', 20, u'New York'],
            [nan, u'Smit', 17, u'Boston'],
            [u'Kalie', u'Flatley', 33, u'Boston'],
            [u'Kittie', u'Schuster', 27, nan],
            [nan, nan, nan, u'South Devyn']
        ],
            columns=['given_name', 'lastname', 'age', 'place'])

        self.A.index.name = 'index_df1'

        self.B = pandas.DataFrame([
            [u'Donel', u'Gerleach', 20, u'New York'],
            [nan, u'Smith', 17, u'Boston'],
            [u'Kaly', u'Flatley', 33, u'Boston'],
            [u'Kittie', nan, 20, nan],
            [u'Bob', u'Armstrong', 70, u'Lake Gavinmouth']
        ],
            columns=['given_name', 'lastname', 'age', 'place'])

        self.B.index.name = 'index_df2'

        self.index_AB = pandas.MultiIndex.from_arrays(
            [arange(len(self.A)), arange(len(self.B))],
            names=[self.A.index.name, self.B.index.name])

##########################################################
#                     EXACT ALGORITHM                    #
##########################################################

    def test_exact(self):

        self.A['test'] = ['Bob', 'Myrthe', 'Ally', 'John', 'Rose']
        self.B['test'] = ['Bob', 'Myrte', 'Ally', 'John', 'Roze']
        expected = pandas.Series([1, 0, 1, 1, 0], index=self.index_AB)

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        result = comp.exact('test', 'test')

        pdt.assert_series_equal(result, expected)

    def test_link_exact_missing(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values as default
        result = comp.exact('given_name', 'given_name')
        expected = pandas.Series([0, 0, 0, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as 0
        result = comp.exact('given_name', 'given_name', missing_value=0)
        expected = pandas.Series([0, 0, 0, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as 9
        result = comp.exact('given_name', 'given_name', missing_value=9)
        expected = pandas.Series([0, 9, 0, 1, 9], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as nan
        result = comp.exact('given_name', 'given_name', missing_value=nan)
        expected = pandas.Series([0, nan, 0, 1, nan], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as string
        result = comp.exact('given_name', 'given_name', missing_value='str')
        expected = pandas.Series([0, 'str', 0, 1, 'str'], index=self.index_AB)
        pdt.assert_series_equal(result, expected)


    def test_link_exact_disagree(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # disagree values as default
        result = comp.exact('given_name', 'given_name')
        expected = pandas.Series([0, 0, 0, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # disagree values as 0
        result = comp.exact('given_name', 'given_name', disagree_value=0)
        expected = pandas.Series([0, 0, 0, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # disagree values as 9
        result = comp.exact('given_name', 'given_name', disagree_value=9)
        expected = pandas.Series([9, 0, 9, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # disagree values as nan
        result = comp.exact('given_name', 'given_name', disagree_value=nan)
        expected = pandas.Series([nan, 0, nan, 1, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # disagree values as string
        result = comp.exact('given_name', 'given_name', disagree_value='str')
        expected = pandas.Series(['str', 0, 'str', 1, 0], index=self.index_AB)

        pdt.assert_series_equal(result, expected)


##########################################################
#                     DATE ALGORITHM                     #
##########################################################

    def test_dates(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        self.A['test_dates'] = pandas.to_datetime(
            ['2005/11/23', np.nan, '2004/11/23', '2010/01/10', '2010/10/30']
        )
        self.B['test_dates'] = pandas.to_datetime(
            ['2005/11/23', '2010/12/31', '2005/11/23', '2010/10/01', '2010/9/30']
        )

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values as default
        result = comp.date('test_dates', 'test_dates')
        expected = pandas.Series([1, 0, 0, 0.5, 0.5], index=self.index_AB)


    def test_dates_with_missings(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        self.A['test_dates'] = pandas.to_datetime(
            ['2005/11/23', np.nan, '2004/11/23', '2010/01/10', '2010/10/30']
        )
        self.B['test_dates'] = pandas.to_datetime(
            ['2005/11/23', '2010/12/31', '2005/11/23', '2010/10/01', '2010/9/30']
        )

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values as default
        print ("Missing values as default")
        result = comp.date('test_dates', 'test_dates')
        expected = pandas.Series([1, 0, 0, 0.5, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as 0
        print ("Missing values as 0")
        result = comp.date('test_dates', 'test_dates', missing_value=0)
        expected = pandas.Series([1, 0, 0, 0.5, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as 123.45
        print ("Missing values as 123.45 (float)")
        result = comp.date('test_dates', 'test_dates', missing_value=123.45)
        expected = pandas.Series([1, 123.45, 0, 0.5, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as nan
        print ("Missing values as numpy.nan")
        result = comp.date('test_dates', 'test_dates', missing_value=nan)
        expected = pandas.Series([1, nan, 0, 0.5, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as string
        print ("Missing values as string")
        result = comp.date('test_dates', 'test_dates', missing_value='str')
        expected = pandas.Series([1, 'str', 0, 0.5, 0.5], index=self.index_AB, dtype=object)
        pdt.assert_series_equal(result, expected)


    def test_dates_with_swap(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        months_to_swap = [
            (9, 10, 123.45),
            (10, 9, 123.45),
            (1, 2, 123.45),
            (2, 1, 123.45)
        ]

        self.A['test_dates'] = pandas.to_datetime(
            ['2005/11/23', np.nan, '2004/11/23', '2010/01/10', '2010/10/30']
        )
        self.B['test_dates'] = pandas.to_datetime(
            ['2005/11/23', '2010/12/31', '2005/11/23', '2010/10/01', '2010/9/30']
        )

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # swap_month_day as default
        print ("swap_month_day and swap_months default")
        result = comp.date('test_dates', 'test_dates')
        expected = pandas.Series([1, 0, 0, 0.5, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # swap_month_day and swap_months as 0
        print ("swap_month_day and swap_months as 0")
        result = comp.date('test_dates', 'test_dates', swap_month_day=0, swap_months='default')
        expected = pandas.Series([1, 0, 0, 0, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # swap_month_day 123.45 (float)
        print ("swap_month_day and swap_months as 123.45 (float)")
        result = comp.date('test_dates', 'test_dates', swap_month_day=123.45, swap_months='default')
        expected = pandas.Series([1, 0, 0, 123.45, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # swap_month_day and swap_months 123.45 (float)
        print ("swap_month_day and swap_months as 123.45 (float)")
        result = comp.date('test_dates', 'test_dates', swap_month_day=123.45, swap_months=months_to_swap)
        expected = pandas.Series([1, 0, 0, 123.45, 123.45], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # swap_month_day and swap_months as nan
        print ("swap_month_day and swap_months as numpy.nan")
        result = comp.date('test_dates', 'test_dates', swap_month_day=nan, swap_months='default', missing_value=nan)
        expected = pandas.Series([1, nan, 0, nan, 0.5], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # swap_month_day as string
        print ("swap_month_day as string")
        result = comp.date('test_dates', 'test_dates', swap_month_day='str')
        expected = pandas.Series([1, 0, 0, 'str', 0.5], index=self.index_AB, dtype=object)
        pdt.assert_series_equal(result, expected)


##########################################################
#                   NUMERIC ALGORITHM                    #
##########################################################

    def test_numeric(self):

        self.A['test_numeric'] = [1, 1, 1, nan, 0]
        self.B['test_numeric'] = [1, 2, 3, nan, nan]

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Basics
        result = comp.numeric('age', 'age', 'step', offset=2)
        expected = pandas.Series([1, 1, 1, 0, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Basics
        result = comp.numeric('age', 'age', method='step', offset=2)
        expected = pandas.Series([1, 1, 1, 0, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Basics
        result = comp.numeric('age', 'age', 'step', 2)
        expected = pandas.Series([1, 1, 1, 0, 0], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

    def test_numeric_with_missings(self):
        """
        Test:
            - Default value
            - numeric value
            - numpy.nan
            - string value
        """

        self.A['test_numeric'] = [1, 1, 1, nan, 1]
        self.B['test_numeric'] = [1, 1, 1, nan, nan]

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values as default
        print ("Missing values as default")
        result = comp.numeric('test_numeric', 'test_numeric', scale=2)
        expected = pandas.Series([1, 1, 1, 0, 0], index=self.index_AB, dtype=np.float64)
        pdt.assert_series_equal(result, expected)

        # Missing values as 0
        print ("Missing values as 0")
        result = comp.numeric('test_numeric', 'test_numeric', scale=2, missing_value=0)
        expected = pandas.Series([1, 1, 1, 0, 0], index=self.index_AB, dtype=np.float64)
        pdt.assert_series_equal(result, expected)

        # Missing values as 123.45
        print ("Missing values as 123.45 (float)")
        result = comp.numeric(
            'test_numeric', 'test_numeric', scale=2, missing_value=123.45)
        expected = pandas.Series([1, 1, 1, 123.45, 123.45], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as nan
        print ("Missing values as numpy.nan")
        result = comp.numeric(
            'test_numeric', 'test_numeric', scale=2, missing_value=nan)
        expected = pandas.Series([1, 1, 1, nan, nan], index=self.index_AB)
        pdt.assert_series_equal(result, expected)

        # Missing values as string
        print ("Missing values as string")
        result = comp.numeric(
            'test_numeric', 'test_numeric', scale=2, missing_value='str')
        expected = pandas.Series([1, 1, 1, 'str', 'str'], index=self.index_AB, dtype=object)
        pdt.assert_series_equal(result, expected)

    def test_numeric_algorithms(self):

        self.A['numeric_val'] = [1, 1, 1, 1, 1]
        self.B['numeric_val'] = [1, 2, 3, 4, 5]

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in NUMERIC_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            # Exclude step algorithm
            if alg is not 'step':
                result = comp.numeric(
                    'numeric_val', 'numeric_val',
                    method=alg, offset=1, scale=2)
            else:
                result = comp.numeric(
                    'numeric_val', 'numeric_val', method=alg, offset=1)

            print (result)


            # All values between 0 and 1.
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

            if alg is not 'step':

                # sim(scale) = 0.5
                expected_bool = pandas.Series(
                    [False, False, False, True, False], index=self.index_AB)
                pdt.assert_series_equal(result == 0.5, expected_bool)

                # sim(offset) = 1
                expected_bool = pandas.Series(
                    [True, True, False, False, False], index=self.index_AB)
                pdt.assert_series_equal(result == 1, expected_bool)

                # sim(scale) larger than 0.5
                expected_bool = pandas.Series(
                    [False, False, True, False, False], index=self.index_AB)
                pdt.assert_series_equal(
                    (result > 0.5) & (result < 1), expected_bool)

                # sim(scale) smaller than 0.5
                expected_bool = pandas.Series(
                    [False, False, False, False, True], index=self.index_AB)
                pdt.assert_series_equal(
                    (result < 0.5) & (result >= 0), expected_bool)

    def test_numeric_alg_errors(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='linear', offset=2, scale=-2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='exp', offset=2, scale=-2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='gauss', offset=2, scale=-2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='squared', offset=2, scale=-2)

        # offset negative
        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='step', offset=-2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='linear', offset=-2, scale=2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='exp', offset=-2, scale=2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='gauss', offset=-2, scale=2)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='squared', offset=-2, scale=2)

    def test_numeric_does_not_exist(self):
        """
        Raise error is the algorithm doesn't exist.
        """

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        with self.assertRaises(ValueError):
            comp.numeric('age', 'age', method='unknown_algorithm')

##########################################################
#                     GEO ALGORITHM                      #
##########################################################

    def test_geo(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values
        result = comp.geo('age', 'age', 'age', 'age', method='linear', offset=2, scale=2)

        self.assertFalse(result.isnull().all())
        # self.assertTrue((result[result.notnull()] >= 0).all())
        # self.assertTrue((result[result.notnull()] <= 1).all())

    def test_geo_batch(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in NUMERIC_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            if alg is not 'step':
                # Missing values
                result = comp.geo('age', 'age', 'age', 'age', method=alg, offset=2, scale=2)
            else:
                result = comp.geo('age', 'age', 'age', 'age', method=alg, offset=2)

            print (result)

            self.assertFalse(result.isnull().all())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_geo_does_not_exist(self):
        """
        Raise error is the algorithm doesn't exist.
        """

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        with self.assertRaises(ValueError):
            comp.geo('age', 'age', 'age', 'age', method='unknown_algorithm')

##########################################################
#                    STRING ALGORITHMS                   #
##########################################################

    def test_fuzzy_same_labels(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in STRING_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            # Missing values
            result = comp.string(
                'given_name', 'given_name', method=alg, missing_value=0)
            result = comp.string(
                'given_name', 'given_name', alg, missing_value=0)

            print (result)

            self.assertFalse(result.isnull().all())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_different_labels(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in STRING_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            # Missing values
            # Change in future (should work without method)
            result = comp.string(
                'given_name', 'given_name', method=alg, missing_value=0)

            print (result)

            self.assertFalse(result.isnull().all())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_errors(self):

        self.A['numeric_value'] = [nan, 2, 3, 4, nan]
        self.B['numeric_value'] = [nan, 2, 3, nan, 5]

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='jaro')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='q_gram')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='cosine')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='jaro_winkler')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='dameraulevenshtein')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='levenshtein')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='smith_waterman')

        with self.assertRaises(Exception):
            comp.string('numeric_value', 'numeric_value', method='lcs')

    def test_fuzzy_nan(self):

        self.A['numeric_value'] = [nan, nan, nan, nan, nan]
        self.B['numeric_value'] = [nan, nan, nan, nan, nan]

        expected = pandas.Series([nan, nan, nan, nan, nan], index=self.index_AB)

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        result = comp.string('numeric_value', 'numeric_value', method='jaro', missing_value=nan)
        pdt.assert_series_equal(result, expected)

        # result = comp.string('numeric_value', 'numeric_value', method='q_gram', missing_value=nan)
        # pdt.assert_series_equal(result, expected)

        # result = comp.string('numeric_value', 'numeric_value', method='cosine', missing_value=nan)
        # pdt.assert_series_equal(result, expected)

        result = comp.string('numeric_value', 'numeric_value', method='jaro_winkler', missing_value=nan)
        pdt.assert_series_equal(result, expected)

        result = comp.string('numeric_value', 'numeric_value', method='dameraulevenshtein', missing_value=nan)
        pdt.assert_series_equal(result, expected)

        result = comp.string('numeric_value', 'numeric_value', method='levenshtein', missing_value=nan)
        pdt.assert_series_equal(result, expected)

        result = comp.string('numeric_value', 'numeric_value', method='lcs', missing_value=nan)
        pdt.assert_series_equal(result, expected)

        result = comp.string('numeric_value', 'numeric_value', method='smith_waterman', missing_value=nan)
        pdt.assert_series_equal(result, expected)

    def test_fuzzy_does_not_exist(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        self.assertRaises(
            ValueError, comp.string, 'given_name',
            'given_name', name='y_name', method='unknown_algorithm')

    def test_smith_waterman_correctness(self):
        self.E = pandas.DataFrame([
            [np.nan],
            [np.nan],
            ['Peter'],
            [''],
            [''],
            ['Peter'],
            ['Peter'],
            ['Anne'],
            ['Elizabeth'],
            ['Sarah'],
            ['University of Waterloo'],
            ['tyler'],
            ['Betty']
        ],
            columns=['str_1'])
        self.E.index.name = 'index_df5'

        self.F = pandas.DataFrame([
            ['Peter'],
            [np.nan],
            [np.nan],
            ['Peter'],
            [''],
            [''],
            ['Peter'],
            ['Jill'],
            ['Elisabeth'],
            ['Sarrrrah'],
            ['University Waterloo'],
            ['Betty'],
            ['tyler']
        ],
            columns=['str_2'])
        self.F.index.name = 'index_df6'

        self.index_EF = pandas.MultiIndex.from_arrays(
            [arange(len(self.E)), arange(len(self.F))],
            names=[self.E.index.name, self.F.index.name])

        comp = recordlinkage.Compare(self.index_EF, self.E, self.F)
        comp.string('str_1', 'str_2', method='smith_waterman', norm='min', name='min_1')
        comp.string('str_1', 'str_2', method='smith_waterman', norm='max', name='max_1')
        comp.string('str_1', 'str_2', method='smith_waterman', norm='mean', name='mean_1')

        comp.string('str_1', 'str_2', method='smith_waterman', norm='min', gap_continue=-5, name='min_2')
        comp.string('str_1', 'str_2', method='smith_waterman', norm='max', gap_continue=-5, name='max_2')
        comp.string('str_1', 'str_2', method='smith_waterman', norm='mean', gap_continue=-5, name='mean_2')

        expected_min_1 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3.6/5, 17.6/19, 2/5, 2/5])
        expected_max_1 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3.6/8, 17.6/22, 2/5, 2/5])
        expected_mean_1 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3.6/6.5, 17.6/20.5, 2/5, 2/5])

        expected_min_2 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3/5, 16/19, 2/5, 2/5])
        expected_max_2 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3/8, 16/22, 2/5, 2/5])
        expected_mean_2 = pandas.Series([0, 0, 0, 0, 0, 0, 1, 0, 7/9, 3/6.5, 16/20.5, 2/5, 2/5])

        SW_TEST_CASES = [
            (comp.vectors['min_1'], expected_min_1, 'min_1'),
            (comp.vectors['max_1'], expected_max_1, 'max_1'),
            (comp.vectors['mean_1'], expected_mean_1, 'mean_1'),
            (comp.vectors['min_2'], expected_min_2, 'min_2'),
            (comp.vectors['max_2'], expected_max_2, 'max_2'),
            (comp.vectors['mean_2'], expected_mean_2, 'mean_2'),
        ]

        for tup in SW_TEST_CASES:
            assert(len(tup[0]) == len(tup[1]))
            for i in range(0, len(tup[0])):
                self.assertAlmostEqual(tup[0].iloc[i], tup[1].iloc[i], places=3, msg='Failed on test {} number {}'.format(tup[2], i))

    def test_lcs_correctness(self):

        self.C = pandas.DataFrame([
            ['peter christen'],
            ['peter christen'],
            ['prap'],
            ['résumé'],
            ['aba'],
        ],
            columns=['str_1'])

        self.C.index.name = 'index_df3'

        self.D = pandas.DataFrame([
            ['christian pedro'],
            ['christen peter'],
            ['papr'],
            ['resume'],
            ['abbaba']
        ],
            columns=['str_2'])

        self.D.index.name = 'index_df4'

        self.index_CD = pandas.MultiIndex.from_arrays(
            [arange(len(self.C)), arange(len(self.D))],
            names=[self.C.index.name, self.D.index.name])

        comp = recordlinkage.Compare(self.index_CD, self.C, self.D)
        comp.string('str_1', 'str_2', method='lcs', norm='dice', min_len=2, name='dice_2')
        comp.string('str_1', 'str_2', method='lcs', norm='jaccard', min_len=2, name='jaccard_2')
        comp.string('str_1', 'str_2', method='lcs', norm='overlap', min_len=2, name='overlap_2')

        comp.string('str_1', 'str_2', method='lcs', norm='dice', min_len=3, name='dice_3')
        comp.string('str_1', 'str_2', method='lcs', norm='jaccard', min_len=3, name='jaccard_3')
        comp.string('str_1', 'str_2', method='lcs', norm='overlap', min_len=3, name='overlap_3')

        expected_dice_2 = pandas.Series([.5517, .9285, .75, .5, .6666])
        expected_jaccard_2 = pandas.Series([.3809, .8666, .6666, .3333, .5])
        expected_overlap_2 = pandas.Series([.5717, .9285, .75, .5, 1])

        expected_dice_3 = pandas.Series([.4137, .9285, 0, .5, .6666])
        expected_jaccard_3 = pandas.Series([.2608, .8666, 0, .3333, .5])
        expected_overlap_3 = pandas.Series([.4285, .9285, 0, .5, 1])

        LCS_TEST_CASES = [
            (comp.vectors['dice_2'], expected_dice_2, 'dice_2'),
            (comp.vectors['jaccard_2'], expected_jaccard_2, 'jaccard_2'),
            (comp.vectors['overlap_2'], expected_overlap_2,'overlap_2'),
            (comp.vectors['dice_3'], expected_dice_3, 'dice_3'),
            (comp.vectors['jaccard_3'], expected_jaccard_3, 'jaccard_3'),
            (comp.vectors['overlap_3'], expected_overlap_3, 'overlap_3'),
        ]

        for tup in LCS_TEST_CASES:
            assert(len(tup[0]) == len(tup[1]))
            for i in range(0, len(tup[0])):
                self.assertAlmostEqual(tup[0].iloc[i], tup[1].iloc[i], places=3, msg='Failed on test {} number {}'.format(tup[2], i))
