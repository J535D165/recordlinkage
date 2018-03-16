#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy
from sklearn.exceptions import NotFittedError

import recordlinkage
import recordlinkage.config as cf

import pytest
from numpy.testing.utils import assert_almost_equal
import pandas.util.testing as ptm


SUPERVISED_CLASSIFIERS = [
    recordlinkage.LogisticRegressionClassifier,
    recordlinkage.NaiveBayesClassifier,
    recordlinkage.SVMClassifier
]

UNSUPERVISED_CLASSIFIERS = [
    recordlinkage.KMeansClassifier
]


class TestClassifyData(object):
    @classmethod
    def setup_class(cls):

        N = 10000
        seed = 101
        seed_gold = 1234  # noqa

        # set random state
        numpy.random.seed(seed)

        cls.y = pandas.DataFrame(
            numpy.random.random((N, 7)),
            index=pandas.MultiIndex.from_arrays(
                [numpy.arange(0, N), numpy.arange(0, N)]),
            columns=[
                'name', 'second_name', 'surname', 'age', 'street', 'state',
                'zipcode'
            ])

        cls.matches_array = numpy.random.randint(0, 2, N)
        cls.matches_series = pandas.Series(
            cls.matches_array, index=cls.y.index)
        cls.matches_index = cls.matches_series[cls.matches_series == 1].index

        cls.y_train = cls.y.iloc[0:1000]


class TestClassifyAPI(TestClassifyData):

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_return_result_options(self, classifier):

        cl = classifier()
        cl.fit(self.y, self.matches_index)

        prediction_default = cl.predict(self.y)
        assert isinstance(prediction_default, pandas.MultiIndex)

        with cf.option_context('classification.return_type', 'index'):
            prediction_multiindex = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_multiindex, pandas.MultiIndex)

        with cf.option_context('classification.return_type', 'array'):
            prediction_ndarray = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_ndarray, numpy.ndarray)

        with cf.option_context('classification.return_type', 'series'):
            prediction_series = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_series, pandas.Series)

        with pytest.raises(ValueError):
            with cf.option_context('classification.return_type',
                                   'unknown_return_type'):
                cl.predict(
                    comparison_vectors=self.y
                )

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_return_result_options_depr(self, classifier):

        cl = classifier()
        cl.fit(self.y, self.matches_index)

        prediction_default = cl.predict(self.y)
        assert isinstance(prediction_default, pandas.MultiIndex)

        prediction_multiindex = cl.predict(
            comparison_vectors=self.y, return_type='index')
        assert isinstance(prediction_multiindex, pandas.MultiIndex)

        prediction_ndarray = cl.predict(
            comparison_vectors=self.y, return_type='array')
        assert isinstance(prediction_ndarray, numpy.ndarray)

        prediction_series = cl.predict(
            comparison_vectors=self.y,
            return_type='series')
        assert isinstance(prediction_series, pandas.Series)

        with pytest.raises(ValueError):
            cl.predict(
                comparison_vectors=self.y,
                return_type='unknown_return_type'
            )

    def test_probs(self):

        cl = recordlinkage.LogisticRegressionClassifier()

        with pytest.raises(ValueError):
            cl.prob(self.y, return_type='unknown_return_type')


class TestKMeansAlgorithms(TestClassifyData):

    def test_kmeans(self):

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.fit(self.y_train)
        result = kmeans.predict(self.y_train)

        assert isinstance(result, pandas.MultiIndex)
        assert result.shape[0] == 519

        kmeans2 = recordlinkage.KMeansClassifier()
        expected = kmeans2.fit_predict(self.y_train)

        assert isinstance(expected, pandas.MultiIndex)

        assert result.values.shape == expected.values.shape
        ptm.assert_index_equal(result, expected)

    def test_kmeans_error(self):

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.fit(self.y_train)

        # There are no probabilities
        with pytest.raises(AttributeError):
            kmeans.prob(self.y)

    def test_kmeans_empty_frame(self):
        """ Kmeans, no training data"""

        kmeans = recordlinkage.KMeansClassifier()

        with pytest.raises(ValueError):
            kmeans.fit(pandas.DataFrame(columns=self.y_train.columns))

    def test_kmeans_not_trained(self):
        """
        Raise an error if the classifier is not trained, but a prediction is
        asked.
        """

        kmeans = recordlinkage.KMeansClassifier()

        with pytest.raises(NotFittedError):
            kmeans.predict(self.y)

    def test_kmeans_manual(self):
        """KMeansClassifier with manual cluster centers"""

        # Make random test data.
        numpy.random.seed(535)
        manual_mcc = list(numpy.random.randn(self.y_train.shape[1]))
        manual_nmcc = list(numpy.random.randn(self.y_train.shape[1]))

        # Initialize the KMeansClassifier
        kmeans = recordlinkage.KMeansClassifier()

        # Check if the cluster centers are None
        assert kmeans.match_cluster_center is None
        assert kmeans.nonmatch_cluster_center is None

        # Set the cluster centers
        kmeans.match_cluster_center = manual_mcc
        kmeans.nonmatch_cluster_center = manual_nmcc

        # Perform the prediction
        kmeans.predict(self.y)

        # Check the match clusters
        mcc = kmeans.match_cluster_center
        nmcc = kmeans.nonmatch_cluster_center
        assert mcc == manual_mcc
        assert nmcc == manual_nmcc


class TestClassifyAlgorithms(TestClassifyData):
    def test_logistic_regression_basic(self):
        """

        Test the LogisticRegressionClassifier by training it, predict on a
        dataset and get the probabilities.

        """

        logis = recordlinkage.LogisticRegressionClassifier()

        # Test the basics
        logis.fit(self.y_train, self.matches_index)
        logis.predict(self.y)
        logis.prob(self.y)

    def test_logistic_regression_manual(self):
        """
        Test the LogisticRegressionClassifier in case of setting the
        parameters manually.

        """

        # Make random test data.
        numpy.random.seed(535)
        manual_coefficients = numpy.random.randn(self.y_train.shape[1])
        manual_intercept = numpy.random.rand()

        # Initialize the LogisticRegressionClassifier
        logis = recordlinkage.LogisticRegressionClassifier()

        # Check if the cofficients and intercapt are None at this point
        assert logis.coefficients is None
        assert logis.intercept is None

        # Set the parameters coefficients and intercept
        logis.coefficients = manual_coefficients
        logis.intercept = manual_intercept

        # Perform the prediction
        logis.predict(self.y)

        # Train the classifier after manula setting
        logis.fit(self.y_train, self.matches_index)
        logis.predict(self.y)

        lc = numpy.array(logis.coefficients)
        assert lc.shape == (self.y_train.shape[1], )
        assert isinstance(logis.intercept, (float))

    def test_bernoulli_naive_bayes(self):
        """Basic Naive Bayes"""

        bernb = recordlinkage.NaiveBayesClassifier()
        bernb.fit(self.y_train.round(), self.matches_index)
        bernb.predict(self.y.round())
        bernb.prob(self.y.round())

    def test_svm(self):

        svm = recordlinkage.SVMClassifier()
        svm.fit(self.y_train, self.matches_index)
        svm.predict(self.y)

        # There are no probabilities
        with pytest.raises(AttributeError):
            svm.prob(self.y)

    def test_em(self):

        ecm = recordlinkage.ECMClassifier()
        ecm.fit(self.y_train.round())
        ecm.predict(self.y.round())
        ecm.prob(self.y.round())

        assert ecm.p is not None
