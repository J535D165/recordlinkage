import pandas
import numpy

import recordlinkage

import pytest


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
    def test_return_result_options(self):

        cl = recordlinkage.Classifier()

        prediction_default = cl._return_result(
            self.matches_array, comparison_vectors=self.y)
        assert isinstance(prediction_default, pandas.MultiIndex)

        prediction_multiindex = cl._return_result(
            self.matches_array, return_type='index', comparison_vectors=self.y)
        assert isinstance(prediction_multiindex, pandas.MultiIndex)

        prediction_ndarray = cl._return_result(
            self.matches_array, return_type='array', comparison_vectors=self.y)
        assert isinstance(prediction_ndarray, numpy.ndarray)

        prediction_series = cl._return_result(
            self.matches_array,
            return_type='series',
            comparison_vectors=self.y)
        assert isinstance(prediction_series, pandas.Series)

        with pytest.raises(ValueError):
            cl._return_result(
                self.matches_array,
                return_type='unknown_return_type',
                comparison_vectors=self.y)

    def test_probs(self):

        cl = recordlinkage.LogisticRegressionClassifier()

        with pytest.raises(ValueError):
            cl.prob(self.y, return_type='unknown_return_type')


class TestClassifyAlgorithms(TestClassifyData):
    def test_kmeans(self):

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.learn(self.y_train)
        kmeans.predict(self.y)

        # There are no probabilities
        with pytest.raises(AttributeError):
            kmeans.prob(self.y)

    def test_kmeans_no_training_data(self):
        """ Kmeans, no training data"""

        kmeans = recordlinkage.KMeansClassifier()

        with pytest.raises(ValueError):
            kmeans.learn(pandas.DataFrame(columns=self.y_train.columns))

    def test_kmeans_not_trained(self):
        """
        Raise an error if the classifier is not trained, but a prediction is
        asked.
        """

        kmeans = recordlinkage.KMeansClassifier()

        with pytest.raises(Exception):
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

    def test_logistic_regression_basic(self):
        """

        Test the LogisticRegressionClassifier by training it, predict on a
        dataset and get the probabilities.

        """

        logis = recordlinkage.LogisticRegressionClassifier()

        # Test the basics
        logis.learn(self.y_train, self.matches_index)
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
        logis.learn(self.y_train, self.matches_index)
        logis.predict(self.y)

        lc = numpy.array(logis.coefficients)
        assert lc.shape == (self.y_train.shape[1], )
        assert isinstance(logis.intercept, (float))

    def test_bernoulli_naive_bayes(self):
        """Basic Naive Bayes"""

        bernb = recordlinkage.NaiveBayesClassifier()
        bernb.learn(self.y_train.round(), self.matches_index)
        bernb.predict(self.y.round())
        bernb.prob(self.y.round())

    def test_svm(self):

        svm = recordlinkage.SVMClassifier()
        svm.learn(self.y_train, self.matches_index)
        svm.predict(self.y)

        # There are no probabilities
        with pytest.raises(AttributeError):
            svm.prob(self.y)

    def test_em(self):

        ecm = recordlinkage.ECMClassifier()
        ecm.learn(self.y_train.round())
        ecm.predict(self.y.round())
        ecm.prob(self.y.round())

        assert ecm.p is not None
