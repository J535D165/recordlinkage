import unittest

import pandas.util.testing as pdt
import recordlinkage

from numpy import nan, arange
import pandas

STRING_SIM_ALGORITHMS = [
    'jaro', 'jaro_winkler', 'dameraulevenshtein',
    'levenshtein', 'q_gram', 'cosine'
]

# nosetests tests/test_compare.py:TestCompare


class TestCompare(unittest.TestCase):

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

# nosetests tests/test_compare.py:TestCompareAPI
class TestCompareAPI(TestCompare):

    def test_instance_linking(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        result = comp.exact('given_name', 'given_name')
        self.assertIsInstance(result, pandas.Series)
        self.assertEqual(result.name, None)

        result = comp.numeric('age', 'age', 2)
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

        result = comp.numeric('age', 'age', 2, name="given_name_comp")
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
        comp.numeric('age', 'age', 2, name="age_comp")
        comp.run()

        self.assertIsInstance(comp.vectors, pandas.DataFrame)
        self.assertEqual(
            comp.vectors.columns.tolist(),
            ["given_name_comp", "lastname_comp", "age_comp"])

        for v in comp.vectors.columns:
            self.assertTrue(comp.vectors[v].notnull().any())


# nosetests tests/test_compare.py:TestCompareAlgorithms
class TestCompareAlgorithms(TestCompare):

    def test_exact(self):

        self.A['test'] = ['Bob', 'Myrthe', 'Ally', 'John', 'Rose']
        self.B['test'] = ['Bob', 'Myrte', 'Ally', 'John', 'Roze']

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        result = comp.exact('test', 'test')
        expected = pandas.Series([1, 0, 1, 1, 0], index=self.index_AB)

        pdt.assert_series_equal(result, expected)

    def test_link_exact_missing(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values as 0
        result = comp.exact('given_name', 'given_name', missing_value=0)
        expected = pandas.Series([0, 0, 0, 1, 0], index=self.index_AB)

        pdt.assert_series_equal(result, expected)

        # Missing values as nan
        result = comp.exact('given_name', 'given_name', missing_value=nan)
        expected = pandas.Series([0, nan, 0, 1, nan], index=self.index_AB)

        pdt.assert_series_equal(result, expected)

        # Missing values as nan
        result = comp.exact('given_name', 'given_name', missing_value=9)
        expected = pandas.Series([0, 9, 0, 1, 9], index=self.index_AB)

        pdt.assert_series_equal(result, expected)

    def test_link_exact_disagree(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values 0 and disagreement as 2
        result = comp.exact('given_name', 'given_name',
                            disagree_value=2, missing_value=0, name='y_name')
        expected = pandas.Series(
            [2, 0, 2, 1, 0], index=self.index_AB, name='y_name')

        pdt.assert_series_equal(result, expected)

    def test_dedup_exact_basic(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        # Missing values
        result = comp.exact('given_name', 'given_name')  # , name='y_name')
        # , name='y_name')
        expected = pandas.Series([1, 0, 1, 1, 0], index=self.index_AB)

        pdt.assert_series_equal(result, expected)

    def test_numeric(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values
        result = comp.numeric('age', 'age', 2)
        expected = pandas.Series(
            [1, 1, 1, 0, 0], index=self.index_AB)  # , name='age')

        pdt.assert_series_equal(result, expected)

    def test_geo(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        # Missing values
        result = comp.geo('age', 'age', 'age', 'age', 2)

        self.assertTrue(result.notnull().any())
        self.assertTrue((result[result.notnull()] >= 0).all())
        self.assertTrue((result[result.notnull()] <= 1).all())

    def test_numeric_batch(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in ['step', 'linear', 'squared']:

            print (alg)

            # Missing values
            result = comp.numeric('age', 'age', 2, method=alg)

            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_does_not_exist(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.A)

        self.assertRaises(
            ValueError, comp.string, 'given_name',
            'given_name', name='y_name', method='unknown_algorithm')

    def test_fuzzy_same_labels(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in STRING_SIM_ALGORITHMS:

            print (alg)

            # Missing values
            result = comp.string('given_name', 'given_name',
                                method=alg, missing_value=0)
            result = comp.string('given_name', 'given_name',
                                alg, missing_value=0)

            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

    def test_fuzzy_different_labels(self):

        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

        for alg in STRING_SIM_ALGORITHMS:

            print (alg)

            # Missing values
            # Change in future (should work without method)
            result = comp.string('given_name', 'given_name',
                                method=alg, missing_value=0)

            print (result)

            self.assertTrue(result.notnull().any())
            self.assertTrue((result[result.notnull()] >= 0).all())
            self.assertTrue((result[result.notnull()] <= 1).all())

            # Debug trick
            # if alg == 'q_gram':
            #     rr

    # def test_batch_compare(self):

    #     comp = recordlinkage.Compare(self.index_AB, self.A, self.B)

    #     # Missing values as 0
    #     result = comp.exact('given_name', 'given_name',
    #                         missing_value=0, name='y_name')

        # pdt.assert_series_equal(result, expected)
