import unittest

import pandas
import numpy

import recordlinkage


# nosetests tests/test_classify.py:TestClassifyData --with-coverage --cover-package=recordlinkage
class TestClassifyData(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        N = 10000
        seed = 101
        seed_gold = 1234

        # set random state
        numpy.random.seed(seed)

        self.y = pandas.DataFrame(
            numpy.random.random((N, 7)),
            index=pandas.MultiIndex.from_arrays(
                [numpy.arange(0, N), numpy.arange(0, N)]),
            columns=[
                'name', 'second_name', 'surname',
                'age', 'street', 'state', 'zipcode'
            ]
        )

        self.matches_array = numpy.random.random_integers(0, 1, N)
        self.matches_series = pandas.Series(self.matches_array)
        self.matches_index = self.matches_series[self.matches_series == 1].index

        self.y_train = self.y.sample(1000, replace=False, random_state=seed_gold)
        self.y_train_index = self.y_train.index


# nosetests tests/test_classify.py:TestClassifyAPI --with-coverage --cover-package=recordlinkage
class TestClassifyAPI(TestClassifyData):

    def test_return_result_options(self):

        cl = recordlinkage.Classifier()

        prediction_default = cl._return_result(
            self.matches_array, comparison_vectors=self.y)
        self.assertTrue(isinstance(prediction_default, pandas.MultiIndex))

        prediction_multiindex = cl._return_result(
            self.matches_array, return_type='index', comparison_vectors=self.y)
        self.assertTrue(isinstance(prediction_multiindex, pandas.MultiIndex))

        prediction_ndarray = cl._return_result(
            self.matches_array, return_type='array', comparison_vectors=self.y)
        self.assertTrue(isinstance(prediction_ndarray, numpy.ndarray))

        prediction_series = cl._return_result(
            self.matches_array, return_type='series', comparison_vectors=self.y)
        self.assertTrue(isinstance(prediction_series, pandas.Series))

        with self.assertRaises(ValueError):
            cl._return_result(
                self.matches_array,
                return_type='unknown_return_type',
                comparison_vectors=self.y
            )


# nosetests tests/test_classify.py:TestClassifyAlgorithms --with-coverage --cover-package=recordlinkage
class TestClassifyAlgorithms(TestClassifyData):

    def test_kmeans(self):

        train_df = self.y.ix[self.match_index].sample(500)
        train_df = train_df.append(self.y.ix[self.y.index.difference(self.match_index)].sample(1500))

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.learn(train_df)
        kmeans.predict(self.y)

    def test_logistic(self):

        train_df = self.y.ix[self.match_index].sample(500)
        train_df = train_df.append(self.y.ix[self.y.index.difference(self.match_index)].sample(1500))

        logis = recordlinkage.LogisticRegressionClassifier()
        logis.learn(train_df, self.match_index)
        logis.predict(self.y)

    def test_bernoulli_naive_bayes(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        bernb = recordlinkage.NaiveBayesClassifier()
        bernb.learn(train_df.round(), train_df_matches.index)
        bernb.predict(self.y.round())

    def test_svm(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        svm = recordlinkage.SVMClassifier()
        svm.learn(train_df.round(), train_df_matches.index)
        svm.predict(self.y.round())

    def test_em(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        ecm = recordlinkage.ECMClassifier()
        ecm.learn(train_df.round())
        ecm.predict(self.y.round())
        ecm.prob(self.y.round())

        self.assertTrue(ecm.p is not None)
