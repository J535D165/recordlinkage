#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

import recordlinkage as rl
import recordlinkage.config as cf

import pytest
from numpy.testing.utils import assert_almost_equal
import pandas.util.testing as ptm


SUPERVISED_CLASSIFIERS = [
    rl.LogisticRegressionClassifier,
    rl.NaiveBayesClassifier,
    rl.SVMClassifier
]

UNSUPERVISED_CLASSIFIERS = [
    rl.KMeansClassifier,
    rl.ECMClassifier
]

CLASSIFIERS = SUPERVISED_CLASSIFIERS + UNSUPERVISED_CLASSIFIERS

CLASSIFIERS_WITH_PROBS = [
    rl.LogisticRegressionClassifier,
    rl.NaiveBayesClassifier
]

N = 10000
Y = pd.DataFrame(
    np.random.random((N, 7)),
    index=pd.MultiIndex.from_arrays(
        [np.arange(0, N), np.arange(0, N)]),
    columns=[
        'name', 'second_name', 'surname', 'age', 'street', 'state',
        'zipcode'
    ])
Y_TRAIN = Y.iloc[0:1000]

MATCHES_ARRAY = np.random.randint(0, 2, N)
MATCHES_SERIES = pd.Series(MATCHES_ARRAY, index=Y.index)
MATCHES_INDEX = MATCHES_SERIES[MATCHES_SERIES == 1].index


class TestClassifyData(object):
    @classmethod
    def setup_class(cls):

        seed = 101
        seed_gold = 1234  # noqa

        # set random state
        np.random.seed(seed)

        cls.y = pd.DataFrame(
            np.random.random((N, 7)),
            index=pd.MultiIndex.from_arrays(
                [np.arange(0, N), np.arange(0, N)]),
            columns=[
                'name', 'second_name', 'surname', 'age', 'street', 'state',
                'zipcode'
            ])

        cls.matches_array = np.random.randint(0, 2, N)
        cls.matches_series = pd.Series(
            cls.matches_array, index=cls.y.index)
        cls.matches_index = cls.matches_series[cls.matches_series == 1].index

        cls.y_train = cls.y.iloc[0:1000]


class TestClassifyAPI(TestClassifyData):

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_return_result_options(self, classifier):

        cl = classifier()
        cl.fit(self.y, self.matches_index)

        prediction_default = cl.predict(self.y)
        assert isinstance(prediction_default, pd.MultiIndex)

        with cf.option_context('classification.return_type', 'index'):
            prediction_multiindex = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_multiindex, pd.MultiIndex)

        with cf.option_context('classification.return_type', 'array'):
            prediction_ndarray = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_ndarray, np.ndarray)

        with cf.option_context('classification.return_type', 'series'):
            prediction_series = cl.predict(comparison_vectors=self.y)
            assert isinstance(prediction_series, pd.Series)

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
        assert isinstance(prediction_default, pd.MultiIndex)

        prediction_multiindex = cl.predict(
            comparison_vectors=self.y, return_type='index')
        assert isinstance(prediction_multiindex, pd.MultiIndex)

        prediction_ndarray = cl.predict(
            comparison_vectors=self.y, return_type='array')
        assert isinstance(prediction_ndarray, np.ndarray)

        prediction_series = cl.predict(
            comparison_vectors=self.y,
            return_type='series')
        assert isinstance(prediction_series, pd.Series)

        with pytest.raises(ValueError):
            cl.predict(
                comparison_vectors=self.y,
                return_type='unknown_return_type'
            )

    @pytest.mark.parametrize('classifier', CLASSIFIERS_WITH_PROBS)
    def test_probs(self, classifier):

        cl = classifier()
        cl.fit(self.y, self.matches_index)

        probs = cl.prob(self.y)

        assert isinstance(probs, pd.Series)
        assert probs.max() <= 1.0
        assert probs.min() >= 0.0

    @pytest.mark.parametrize('classifier', UNSUPERVISED_CLASSIFIERS)
    def test_fit_predict_unsupervised(self, classifier):

        cl = classifier()
        cl.fit(self.y_train)
        result = cl.predict(self.y_train)

        assert isinstance(result, pd.MultiIndex)

        cl2 = classifier()
        expected = cl2.fit_predict(self.y_train)

        assert isinstance(expected, pd.MultiIndex)
        assert result.values.shape == expected.values.shape

        ptm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_fit_predict_supervised(self, classifier):

        cl = classifier()
        cl.fit(self.y_train, self.matches_index)
        result = cl.predict(self.y_train)

        assert isinstance(result, pd.MultiIndex)

        cl2 = classifier()
        expected = cl2.fit_predict(self.y_train, self.matches_index)

        assert isinstance(expected, pd.MultiIndex)
        assert result.values.shape == expected.values.shape

        ptm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('classifier', CLASSIFIERS)
    def test_predict_but_not_trained(self, classifier):

        cl = classifier()

        with pytest.raises(NotFittedError):
            cl.predict(self.y)

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_fit_empty_frame_supervised(self, classifier):

        cl = classifier()

        with pytest.raises(ValueError):
            cl.fit(
                pd.DataFrame(columns=self.y_train.columns),
                self.matches_index
            )

    @pytest.mark.parametrize('classifier', UNSUPERVISED_CLASSIFIERS)
    def test_fit_empty_frame_unsupervised(self, classifier):

        cl = classifier()

        with pytest.raises(ValueError):
            cl.fit(pd.DataFrame(columns=self.y_train.columns))


class TestKMeansAlgorithms(TestClassifyData):

    def test_kmeans(self):

        kmeans = rl.KMeansClassifier()
        kmeans.fit(self.y_train)
        result = kmeans.predict(self.y_train)

        assert isinstance(result, pd.MultiIndex)
        assert result.shape[0] == 519

    def test_kmeans_error(self):

        kmeans = rl.KMeansClassifier()
        kmeans.fit(self.y_train)

        # There are no probabilities
        with pytest.raises(AttributeError):
            kmeans.prob(self.y)

    def test_kmeans_manual(self):
        """KMeansClassifier with manual cluster centers"""

        # Make random test data.
        np.random.seed(535)
        manual_mcc = list(np.random.randn(self.y_train.shape[1]))
        manual_nmcc = list(np.random.randn(self.y_train.shape[1]))

        # Initialize the KMeansClassifier
        kmeans = rl.KMeansClassifier()

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

        logis = rl.LogisticRegressionClassifier()

        # Test the basics
        logis.fit(self.y_train, self.matches_index)
        logis.predict(self.y)
        logis.prob(self.y)

    def test_logistic_regression_manual(self):

        # Make random test data.
        np.random.seed(535)
        manual_coefficients = np.random.randn(self.y_train.shape[1])
        manual_intercept = np.random.rand()

        # Initialize the LogisticRegressionClassifier
        logis = rl.LogisticRegressionClassifier()

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

        lc = np.array(logis.coefficients)
        assert lc.shape == (self.y_train.shape[1], )
        assert isinstance(logis.intercept, (float))

    def test_svm(self):

        svm = rl.SVMClassifier()
        svm.fit(self.y_train, self.matches_index)
        svm.predict(self.y)

        # There are no probabilities
        with pytest.raises(AttributeError):
            svm.prob(self.y)


class TestClassifyECM(TestClassifyData):

    def test_ecm_probs(self):

        ecm = rl.ECMClassifier()
        ecm.fit(self.y_train.round())

        assert (ecm.p <= 1.0) & (ecm.p >= 0.0)

    def test_ecm_predict(self):

        ecm = rl.ECMClassifier()
        ecm.fit(self.y.round())
        prediction = ecm.predict(self.y.round())


class TestFellegiSunter(TestClassifyData):

    @pytest.mark.parametrize('classifier, fit_args', [
        (rl.NaiveBayesClassifier, (Y_TRAIN, MATCHES_INDEX)),
        (rl.ECMClassifier, (Y_TRAIN,))
    ])
    def test_FS_parameters(self, classifier, fit_args):

        cl = classifier()
        cl.fit(*fit_args)

        # p
        assert np.isscalar(cl.p)
        assert np.exp(cl.log_p) == cl.p

        # m
        assert isinstance(cl.m_probs, np.ndarray)
        assert cl.m_probs.shape == (Y_TRAIN.shape[1],)
        assert_almost_equal(np.exp(cl.log_m_probs), cl.m_probs)

        # u
        assert isinstance(cl.u_probs, np.ndarray)
        assert cl.u_probs.shape == (Y_TRAIN.shape[1],)
        assert_almost_equal(np.exp(cl.log_u_probs), cl.u_probs)

    # def test_FS_supervised_binarize(self):

    #     cl = rl.NaiveBayesClassifier(binarize=None)
    #     cl.fit(self.y_train, self.matches_index)
    #     cl.predict(self.y)
    #     cl.prob(self.y)
