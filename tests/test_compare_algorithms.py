import unittest

import pandas.util.testing as pdt
import numpy.testing as npt
import recordlinkage

import numpy as np
from numpy import nan, arange
import pandas

from test_compare import TestCompare

STRING_SIM_ALGORITHMS = [
    'jaro', 'q_gram', 'cosine', 'jaro_winkler', 'dameraulevenshtein', 'levenshtein'
]

NUMERIC_SIM_ALGORITHMS = [
    'step', 'linear', 'squared', 'exp', 'gauss'
]

# nosetests tests/test_compare_algorithms.py:TestCompareAlgorithms
class TestCompareAlgorithms(TestCompare):

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

        for alg in NUMERIC_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            if alg is not 'step':
                with self.assertRaises(ValueError):
                    yield comp.numeric('age', 'age', method=alg, offset=-2, scale=2)

                with self.assertRaises(ValueError):
                    yield comp.numeric('age', 'age', method=alg, offset=2, scale=-2)

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

        self.A['numeric_value'] = [1, 2, 3, 4, 5]
        self.B['numeric_value'] = [1, 2, 3, 4, 5]

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in STRING_SIM_ALGORITHMS:

            print ('The {} algorithm'.format(alg))

            with self.assertRaises(Exception):
                # Missing values
                comp.string('numeric_value', 'numeric_value', method=alg, missing_value=0)

    def test_fuzzy_does_not_exist(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        self.assertRaises(
            ValueError, comp.string, 'given_name',
            'given_name', name='y_name', method='unknown_algorithm')
