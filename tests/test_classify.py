#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np
from numpy.testing.utils import assert_almost_equal

import pandas as pd
import pandas.util.testing as ptm

import pytest

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer

import recordlinkage as rl
from recordlinkage.datasets import binary_vectors

SUPERVISED_CLASSIFIERS = [
    rl.LogisticRegressionClassifier,
    rl.NaiveBayesClassifier,
    rl.SVMClassifier
]

UNSUPERVISED_CLASSIFIERS = [
    rl.KMeansClassifier,
    rl.ECMClassifier
]

CLASSIFIERS_WITH_PROBS = [
    rl.LogisticRegressionClassifier,
    rl.NaiveBayesClassifier,
    rl.ECMClassifier
]

CLASSIFIERS = SUPERVISED_CLASSIFIERS + UNSUPERVISED_CLASSIFIERS


N = 10000


class TestClassifyData(object):
    @classmethod
    def setup_class(cls):

        cls.render_bin_test_data()

    @classmethod
    def render_bin_test_data(cls, n_pairs_train=5000, n_matches_train=1000,
                             n_pairs_test=50000, n_matches_test=10000):

        cls.m = np.array([.92, .81, .85, .90, .99, .70, .56])
        cls.u = np.array([.19, .23, .50, .11, .20, .14, .50])

        cls.labels = [
            'name',
            'second_name',
            'surname',
            'dob',
            'street',
            'state',
            'zipcode'
        ]

        # Create the train dataset.
        cls.X_train, cls.y_train = binary_vectors(
            n_pairs_train,
            n_matches_train,
            m=cls.m,
            u=cls.u,
            random_state=535,
            return_links=True)

        cls.X_train.columns = cls.labels

        # Create the test dataset.
        cls.X_test, cls.y_test = binary_vectors(
            n_pairs_test,
            n_matches_test,
            m=cls.m,
            u=cls.u,
            random_state=535,
            return_links=True)

        cls.y_test.columns = cls.labels


class TestClassifyAPI(TestClassifyData):

    @pytest.mark.parametrize('classifier', CLASSIFIERS)
    def test_return_result_options(self, classifier):

        cl = classifier()
        if isinstance(cl, tuple(UNSUPERVISED_CLASSIFIERS)):
            cl.fit(self.X_train)
        else:
            cl.fit(self.X_train, self.y_train)

        prediction_default = cl.predict(self.X_test)
        assert isinstance(prediction_default, pd.MultiIndex)

        with rl.option_context('classification.return_type', 'index'):
            prediction_multiindex = cl.predict(comparison_vectors=self.X_train)
            assert isinstance(prediction_multiindex, pd.MultiIndex)

        with rl.option_context('classification.return_type', 'array'):
            prediction_ndarray = cl.predict(comparison_vectors=self.X_train)
            assert isinstance(prediction_ndarray, np.ndarray)

        with rl.option_context('classification.return_type', 'series'):
            prediction_series = cl.predict(comparison_vectors=self.X_train)
            assert isinstance(prediction_series, pd.Series)

        with pytest.raises(ValueError):
            with rl.option_context('classification.return_type',
                                   'unknown_return_type'):
                cl.predict(
                    comparison_vectors=self.X_train
                )

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_return_result_options_depr(self, classifier):

        cl = classifier()
        cl.fit(self.X_train, self.y_train)

        prediction_default = cl.predict(self.X_test)
        assert isinstance(prediction_default, pd.MultiIndex)

        with pytest.deprecated_call():
            prediction_multiindex = cl.predict(
                comparison_vectors=self.X_train, return_type='index')
            assert isinstance(prediction_multiindex, pd.MultiIndex)

        with pytest.deprecated_call():
            prediction_ndarray = cl.predict(
                comparison_vectors=self.X_train, return_type='array')
            assert isinstance(prediction_ndarray, np.ndarray)

        with pytest.deprecated_call():
            prediction_series = cl.predict(
                comparison_vectors=self.X_train,
                return_type='series')
            assert isinstance(prediction_series, pd.Series)

        with pytest.deprecated_call():
            with pytest.raises(ValueError):
                cl.predict(
                    comparison_vectors=self.X_train,
                    return_type='unknown_return_type'
                )

    @pytest.mark.parametrize('classifier', CLASSIFIERS_WITH_PROBS)
    def test_probs(self, classifier):

        cl = classifier()

        if isinstance(cl, tuple(UNSUPERVISED_CLASSIFIERS)):
            cl.fit(self.X_train)
        else:
            cl.fit(self.X_train, self.y_train)

        probs = cl.prob(self.X_test)
        print(probs)

        assert isinstance(probs, pd.Series)
        assert probs.notnull().all()
        assert probs.max() <= 1.0
        assert probs.min() >= 0.0

    @pytest.mark.parametrize('classifier', UNSUPERVISED_CLASSIFIERS)
    def test_fit_predict_unsupervised(self, classifier):

        cl = classifier()
        cl.fit(self.X_train)
        result = cl.predict(self.X_train)

        assert isinstance(result, pd.MultiIndex)

        cl2 = classifier()
        expected = cl2.fit_predict(self.X_train)

        assert isinstance(expected, pd.MultiIndex)
        assert result.values.shape == expected.values.shape

        ptm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_fit_predict_supervised(self, classifier):

        cl = classifier()
        cl.fit(self.X_train, self.y_train)
        result = cl.predict(self.X_train)

        assert isinstance(result, pd.MultiIndex)

        cl2 = classifier()
        expected = cl2.fit_predict(self.X_train, self.y_train)

        assert isinstance(expected, pd.MultiIndex)
        assert result.values.shape == expected.values.shape

        ptm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('classifier', CLASSIFIERS)
    def test_predict_but_not_trained(self, classifier):

        cl = classifier()

        with pytest.raises(NotFittedError):
            cl.predict(self.X_test)

    @pytest.mark.parametrize('classifier', SUPERVISED_CLASSIFIERS)
    def test_fit_empty_frame_supervised(self, classifier):

        cl = classifier()

        with pytest.raises(ValueError):
            cl.fit(
                pd.DataFrame(columns=self.X_train.columns),
                self.y_train
            )

    @pytest.mark.parametrize('classifier', UNSUPERVISED_CLASSIFIERS)
    def test_fit_empty_frame_unsupervised(self, classifier):

        cl = classifier()

        with pytest.raises(ValueError):
            cl.fit(pd.DataFrame(columns=self.X_train.columns))


class TestKMeans(TestClassifyData):

    def test_kmeans(self):

        kmeans = rl.KMeansClassifier()
        kmeans.fit(self.X_train)
        result = kmeans.predict(self.X_test)

        assert isinstance(result, pd.MultiIndex)
        assert result.shape[0] == 11670

    def test_kmean_parameters(self):

        kmeans = rl.KMeansClassifier()
        kmeans.fit(self.X_train)

        _, n_features = self.X_train.shape

        assert isinstance(kmeans.match_cluster_center, np.ndarray)
        assert kmeans.match_cluster_center.shape == (n_features,)

        assert isinstance(kmeans.nonmatch_cluster_center, np.ndarray)
        assert kmeans.nonmatch_cluster_center.shape == (n_features,)

    def test_kmeans_error(self):

        kmeans = rl.KMeansClassifier()
        kmeans.fit(self.X_train)

        # There are no probabilities
        with pytest.raises(AttributeError):
            kmeans.prob(self.X_train)

    def test_kmeans_manual(self):
        """KMeansClassifier with manual cluster centers"""

        # Make random test data.
        np.random.seed(535)
        manual_mcc = list(np.random.randn(self.X_train.shape[1]))
        manual_nmcc = list(np.random.randn(self.X_train.shape[1]))

        # Initialize the KMeansClassifier
        kmeans = rl.KMeansClassifier()

        # Check if the cluster centers are None
        assert not hasattr(kmeans, 'match_cluster_center')
        assert not hasattr(kmeans, 'nonmatch_cluster_center')

        # Set the cluster centers
        kmeans.match_cluster_center = manual_mcc
        kmeans.nonmatch_cluster_center = manual_nmcc

        # Perform the prediction
        kmeans.predict(self.X_test)

        # Check the match clusters
        mcc = kmeans.match_cluster_center
        nmcc = kmeans.nonmatch_cluster_center
        assert_almost_equal(mcc, manual_mcc)
        assert_almost_equal(nmcc, manual_nmcc)


class TestLogistic(TestClassifyData):
    def test_logistic_regression_basic(self):

        logis = rl.LogisticRegressionClassifier()

        # Test the basics
        logis.fit(self.X_train, self.y_train)
        logis.predict(self.X_test)
        logis.prob(self.X_train)

    def test_logistic_regression_manual(self):

        # Make random test data.
        np.random.seed(535)
        manual_coefficients = np.random.randn(self.X_train.shape[1])
        manual_intercept = np.random.rand()

        # Initialize the LogisticRegressionClassifier
        logis = rl.LogisticRegressionClassifier()
        assert not hasattr(logis, 'coefficients')
        assert not hasattr(logis, 'intercept')

        # Set the parameters coefficients and intercept
        logis.coefficients = manual_coefficients
        logis.intercept = manual_intercept

        # Perform the prediction
        logis.predict(self.X_test)

        # Train the classifier after manual setting
        logis.fit(self.X_train, self.y_train)
        logis.predict(self.X_test)

        lc = logis.coefficients
        assert lc.shape == (self.X_train.shape[1], )
        assert isinstance(logis.intercept, (float))


class TestSVM(TestClassifyData):

    def test_svm(self):

        svm = rl.SVMClassifier()
        svm.fit(self.X_train, self.y_train)
        svm.predict(self.X_test)

        # There are no probabilities
        with pytest.raises(AttributeError):
            svm.prob(self.X_train)


class TestECM(TestClassifyData):
    """Test ECM Classifier"""

    def test_sklearn_labelbin(self):

        m = np.array([1.0, .81, .85, .81, .85, .81])
        u = np.array([1.0, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        binarizer = LabelBinarizer()
        binarizer.fit(X_train.iloc[:, 0])
        assert len(binarizer.classes_) == 1

        binarizer.classes_ = np.array([0, 1])
        assert len(binarizer.classes_) == 2

        binarizer.transform(X_train.iloc[:, 1])
        assert len(binarizer.classes_) == 2

    def test_sklearn_preinit(self):

        m = np.array([1.0, .81, .85, .81, .85, .81])
        u = np.array([1.0, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        binarizer = LabelBinarizer()
        binarizer.classes_ = np.array([0, 1])

        binarizer.transform(X_train.iloc[:, 1])
        assert len(binarizer.classes_) == 2

    def test_ecm_probs(self):

        ecm = rl.ECMClassifier()
        ecm.fit(self.X_train.round())

        assert (ecm.p <= 1.0) & (ecm.p >= 0.0)

    def test_ecm_predict(self):

        ecm = rl.ECMClassifier()
        ecm.fit(self.X_train.round())
        ecm.predict(self.X_test)

    def test_ecm_init(self):

        m = np.array([0.23, .81, .85, .81, .85, .81])
        u = np.array([0.34, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        ecm = rl.ECMClassifier(init='random')
        ecm.fit(X_train)
        ecm.predict(X_train)

        print(ecm.m_probs)
        print(ecm.log_m_probs)
        print(ecm.u_probs)
        print(ecm.log_u_probs)

        assert math.isclose(ecm.m_probs['c_2'][1], 0.85, abs_tol=0.08)

    def test_ecm_init_random_1value(self):

        m = np.array([1.0, .81, .85, .81, .85, .81])
        u = np.array([1.0, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=536, return_links=True)

        ecm = rl.ECMClassifier(init='random')
        ecm.fit(X_train)
        ecm.predict(X_train)

        with pytest.raises(KeyError):
            ecm.m_probs['c_1'][0]

        assert math.isclose(ecm.m_probs['c_2'][1], 0.85, abs_tol=0.08)
        assert math.isclose(ecm.p, 0.5, abs_tol=0.05)

    def test_ecm_init_jaro_1value(self):

        m = np.array([1.0, 0.85, .85, .81, .85, .81])
        u = np.array([1.0, .10, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        ecm = rl.ECMClassifier(init='jaro')
        ecm.fit(X_train)
        ecm.predict(X_train)

        with pytest.raises(KeyError):
            ecm.m_probs['c_1'][0]

        assert math.isclose(ecm.m_probs['c_1'][1], 1.0, abs_tol=0.01)
        assert math.isclose(ecm.m_probs['c_2'][1], 0.85, abs_tol=0.08)
        assert math.isclose(ecm.u_probs['c_1'][1], 1.0, abs_tol=0.01)
        assert math.isclose(ecm.u_probs['c_2'][1], 0.1, abs_tol=0.05)
        assert math.isclose(ecm.p, 0.5, abs_tol=0.05)

    def test_ecm_init_jaro_skewed(self):

        m = np.array([1.0, 0.85, .85, .81, .85, .81])
        u = np.array([0.0, .10, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        ecm = rl.ECMClassifier(init='jaro')
        ecm.fit(X_train)
        ecm.predict(X_train)

        assert math.isclose(ecm.m_probs['c_1'][1], 1.0, abs_tol=0.01)
        assert math.isclose(ecm.m_probs['c_2'][1], 0.85, abs_tol=0.08)
        assert math.isclose(ecm.u_probs['c_1'][1], 0.0, abs_tol=0.01)
        assert math.isclose(ecm.u_probs['c_2'][1], 0.1, abs_tol=0.05)
        assert math.isclose(ecm.p, 0.5, abs_tol=0.05)

    def test_ecm_init_jaro_inf(self):
        m = np.array([0.95, .81, .85, .81, .85, .81])
        u = np.array([0, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            10000, 500, m=m, u=u, random_state=535, return_links=True)

        # Create the train dataset.
        X_test, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        ecm = rl.ECMClassifier()
        ecm.fit(X_train)
        ecm.predict(X_test)

        assert math.isclose(ecm.u_probs['c_1'][1], 0.0, abs_tol=1e-3)
        assert math.isclose(ecm.u_probs['c_1'][0], 1.0, abs_tol=1e-3)

    def test_binary_input(self):
        m = np.array([1, .81, .85, .81, .85, .81])
        u = np.array([1, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            5000, 500, m=m, u=u, random_state=535, return_links=True)

        # Create the train dataset.
        X_test, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        ecm = rl.ECMClassifier()
        ecm.fit(X_train)
        ecm.predict(X_test)

    def test_binarize_input(self):
        m = np.array([1, .81, .85, .81, .85, .81])
        u = np.array([1, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)
        X_train = X_train * np.random.rand(*X_train.shape)

        # Create the train dataset.
        X_test, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)
        X_test = X_test * np.random.rand(*X_test.shape)

        ecm = rl.ECMClassifier(binarize=True)
        ecm.fit(X_train)
        ecm.predict(X_test)


class TestFellegiSunter(TestClassifyData):

    @pytest.mark.parametrize('classifier', [
        rl.NaiveBayesClassifier,
        rl.ECMClassifier
    ])
    def test_fs_parameters(self, classifier):

        cl = classifier()
        if isinstance(cl, tuple(UNSUPERVISED_CLASSIFIERS)):
            cl.fit(self.X_train)
        else:
            cl.fit(self.X_train, self.y_train)

        # p
        assert np.isscalar(cl.p)
        assert np.exp(cl.log_p) == cl.p

        # m
        assert isinstance(cl.m_probs, dict)
        assert len(cl.m_probs.keys()) == self.X_train.shape[1]
        for col, value in cl.m_probs.items():
            for key, out in value.items():
                assert_almost_equal(
                    np.exp(cl.log_m_probs[col][key]),
                    cl.m_probs[col][key]
                )

        # u
        assert isinstance(cl.u_probs, dict)
        assert len(cl.u_probs.keys()) == self.X_train.shape[1]
        for col, value in cl.u_probs.items():
            for key, out in value.items():
                assert_almost_equal(
                    np.exp(cl.log_u_probs[col][key]),
                    cl.u_probs[col][key]
                )

    @pytest.mark.parametrize('classifier', [
        rl.NaiveBayesClassifier,
        rl.ECMClassifier
    ])
    def test_fs_column_labels(self, classifier):

        m = np.array([0.95, .81, .85, .81, .85, .81])
        u = np.array([0, .23, .50, .23, .30, 0.13])

        # Create the train dataset.
        X_train, true_links = binary_vectors(
            1000, 500, m=m, u=u, random_state=535, return_links=True)

        cl = classifier()
        if isinstance(cl, tuple(UNSUPERVISED_CLASSIFIERS)):
            cl.fit(X_train)
        else:
            cl.fit(X_train, true_links)

        assert set([*cl.m_probs]) == set(list(X_train))
        assert set([*cl.u_probs]) == set(list(X_train))
        assert set([*cl.log_m_probs]) == set(list(X_train))
        assert set([*cl.log_m_probs]) == set(list(X_train))

    # @pytest.mark.parametrize('classifier', [
    #     rl.NaiveBayesClassifier,
    #     rl.ECMClassifier
    # ])
    # def test_fs_parameters_set_get(self, classifier):

    #     # there were some issues with setting and getting parameters. Afters
    #     # getting parameters, the internel parameters were messed up.

    #     cl = classifier()
    #     if isinstance(cl, tuple(UNSUPERVISED_CLASSIFIERS)):
    #         cl.fit(Y_TRAIN)
    #     else:
    #         cl.fit(Y_TRAIN, MATCHES_INDEX)

    #     probs_before = cl.prob(Y_TRAIN)
    #     predict_before = cl.predict(Y_TRAIN)

    #     # p
    #     attributes = ["p", "log_p",
    #                   "m_probs", "u_probs",
    #                   "log_m_probs", "log_u_probs",
    #                   "weights", "log_weights"]

    #     for attr in attributes:
    #         print(attr)
    #         value = getattr(cl, attr)

    #         if attr not in ["weights", "log_weights"]:
    #             setattr(cl, attr, value)

    #     probs_after = cl.prob(Y_TRAIN)
    #     predict_after = cl.predict(Y_TRAIN)

    #     ptm.assert_series_equal(probs_before, probs_after)
    #     ptm.assert_index_equal(predict_before, predict_after)

    # def test_FS_supervised_binarize(self):

    #     cl = rl.NaiveBayesClassifier(binarize=None)
    #     cl.fit(self.X_train, self.y_train)
    #     cl.predict(self.X_test)
    #     cl.prob(self.X_train)
