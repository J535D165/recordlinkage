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
        self.matches_series = pandas.Series(self.matches_array, index=self.y.index)
        self.matches_index = self.matches_series[self.matches_series == 1].index

        self.y_train = self.y.iloc[0:1000]


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

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.learn(self.y_train)
        kmeans.predict(self.y)

    def test_kmeans_no_training_data(self):

        kmeans = recordlinkage.KMeansClassifier()

        with self.assertRaises(ValueError):
            kmeans.learn(pandas.DataFrame(columns=self.y_train.columns))

    def test_kmeans_not_trained(self):
        """
        Raise an error if the classifier is not trained, but a prediction is
        asked.
        """

        kmeans = recordlinkage.KMeansClassifier()

        with self.assertRaises(Exception):
            kmeans.predict(self.y)

    def test_logistic(self):

        logis = recordlinkage.LogisticRegressionClassifier()
        logis.learn(self.y_train, self.matches_index)
        logis.predict(self.y)
        # logis.prob(self.y)

    def test_logistic_advanced(self):

        logis = recordlinkage.LogisticRegressionClassifier()
        logis.learn(self.y_train, self.matches_index)
        logis.predict(self.y)
        # logis.prob(self.y)

    def test_bernoulli_naive_bayes(self):

        bernb = recordlinkage.NaiveBayesClassifier()
        bernb.learn(self.y_train.round(), self.matches_index)
        bernb.predict(self.y.round())

    def test_svm(self):

        svm = recordlinkage.SVMClassifier()
        svm.learn(self.y_train, self.matches_index)
        svm.predict(self.y.round())

    def test_em(self):

        ecm = recordlinkage.ECMClassifier()
        ecm.learn(self.y_train.round())
        ecm.predict(self.y.round())
        ecm.prob(self.y.round())

        self.assertTrue(ecm.p is not None)
