import time

import pandas
import numpy

from sklearn import cluster, linear_model, naive_bayes, svm

from recordlinkage.base import BaseClassifier as Classifier
from recordlinkage.algorithms.em import ECMEstimate

from recordlinkage import rl_logging as logging


class KMeansClassifier(Classifier):
    """KMeans classifier.

    The `K-means clusterings algorithm (wikipedia)
    <https://en.wikipedia.org/wiki/K-means_clustering>`_ partitions candidate
    record pairs into matches and non-matches. Each comparison vector belongs
    to the cluster with the nearest mean. The K-means algorithm does not need
    trainings data, but it needs two starting points (one for the matches and
    one for the non-matches). The K-means clustering problem is NP-hard.

    Parameters
    ----------
    match_cluster_center : list, numpy.array
        The center of the match cluster. The length of the list/array must
        equal the number of comparison variables.
    nonmatch_cluster_center : list, numpy.array
        The center of the nonmatch (distinct) cluster. The length of the
        list/array must equal the number of comparison variables.

    Attributes
    ----------
    classifier: sklearn.cluster.KMeans
        The Kmeans cluster class in sklearn.
    match_cluster_center : list, numpy.array
        The center of the match cluster.
    nonmatch_cluster_center : list, numpy.array
        The center of the nonmatch (distinct) cluster.

    Note
    ----
    There are way better methods for linking records than the k-means
    clustering algorithm. However, this algorithm does not need trainings data
    and is useful to do an initial partition.
    """

    def __init__(self, match_cluster_center=None,
                 nonmatch_cluster_center=None, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        # initialize the classifier
        self.classifier = cluster.KMeans(n_clusters=2, n_init=1)

        self._match_cluster_center = match_cluster_center
        self._nonmatch_cluster_center = nonmatch_cluster_center

        # set cluster centers if available
        self._set_cluster_centers(
            self._match_cluster_center, self._nonmatch_cluster_center)

    @property
    def match_cluster_center(self):
        # Return the centers if available
        try:
            return self.classifier.cluster_centers_.tolist()[1]
        except AttributeError:
            return None

    @match_cluster_center.setter
    def match_cluster_center(self, value):

        self._match_cluster_center = value

        try:
            self._set_cluster_centers(
                self._match_cluster_center, self._nonmatch_cluster_center)
        except ValueError:
            pass

    @property
    def nonmatch_cluster_center(self):
        # Return the centers if available
        try:
            return self.classifier.cluster_centers_.tolist()[0]
        except AttributeError:
            return None

    @nonmatch_cluster_center.setter
    def nonmatch_cluster_center(self, value):

        self._nonmatch_cluster_center = value

        try:
            self._set_cluster_centers(
                self._match_cluster_center, self._nonmatch_cluster_center)
        except ValueError:
            pass

    def _set_cluster_centers(self, m_cluster_center, n_cluster_center):

        if m_cluster_center is not None and n_cluster_center is not None:
            self.classifier.cluster_centers_ = numpy.array(
                [numpy.array(n_cluster_center), numpy.array(m_cluster_center)]
            )
        elif m_cluster_center is None and n_cluster_center is None:
            try:
                del self.classifier.cluster_centers_
            except AttributeError:
                pass
        else:
            raise ValueError("set the center of the match cluster and the " +
                             "nonmatch cluster")

    def learn(self, comparison_vectors, return_type='index'):
        """Train the K-means classifier.

        The K-means classifier is unsupervised and therefore does not need
        labels. The K-means classifier classifies the data into two sets of
        links and non- links. The starting point of the cluster centers are
        0.05 for the non-matches and 0.95 for the matches.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The dataframe with comparison vectors.
        return_type : 'index' (default), 'series', 'array'
            The format to return the classification result. The argument value
            'index' will return the pandas.MultiIndex of the matches. The
            argument value 'series' will return a pandas.Series with zeros
            (distinct) and ones (matches). The argument value 'array' will
            return a numpy.ndarray with zeros and ones.

        Returns
        -------
        pandas.MultiIndex, pandas.Series or numpy.ndarray
            The prediction (see also the argument 'return_type')

        """

        # Set the start point of the classifier.
        self.classifier.init = numpy.array([
            [0.05] * len(list(comparison_vectors)),
            [0.95] * len(list(comparison_vectors))
        ])

        # Fit and predict
        prediction = self.classifier.fit_predict(
            comparison_vectors.as_matrix())

        return self._return_result(prediction, return_type, comparison_vectors)

    def prob(self, *args, **kwargs):

        raise AttributeError(
            "It is not possible to compute "
            "probabilities for the KMeansClassfier")


# DeterministicClassifier = LogisticRegressionClassifier
class LogisticRegressionClassifier(Classifier):
    """Logistic Regression Classifier.

    This classifier is an application of the `logistic regression model
    (wikipedia) <https://en.wikipedia.org/wiki/Logistic_regression>`_. The
    classifier partitions candidate record pairs into matches and non-matches.

    Parameters
    ----------
    coefficients : list, numpy.array
        The coefficients of the logistic regression.
    intercept : float
        The interception value.

    Attributes
    ----------
    classifier: sklearn.linear_model.LogisticRegression
        The Logistic regression classifier in sklearn.
    coefficients : list
        The coefficients of the logistic regression.
    intercept : float
        The interception value.

    """

    def __init__(self, coefficients=None, intercept=None):
        super(self.__class__, self).__init__()

        self.classifier = linear_model.LogisticRegression()

        self.coefficients = coefficients
        self.intercept = intercept

        self.classifier.classes_ = numpy.array([False, True])

    @property
    def params(self):
        return {
            'coefficients': self.coefficients,
            'intercept': self.intercept
        }

    @params.setter
    def params(self, value):

        if not isinstance(value, dict):
            raise ValueError("parameters are of wrong type")

        self.coefficients = value['coefficients']
        self.intercept = value['intercept']

    @property
    def coefficients(self):
        # Return the coefficients if available
        try:
            return list(self.classifier.coef_[0])
        except Exception:
            return None

    @coefficients.setter
    def coefficients(self, value):

        if value is not None:

            # Check if array if numpy array
            if type(value) is not numpy.ndarray:
                value = numpy.array(value)

            self.classifier.coef_ = value.reshape((1, len(value)))

    @property
    def intercept(self):

        try:
            return float(self.classifier.intercept_[0])
        except Exception:
            return None

    @intercept.setter
    def intercept(self, value):

        if value is not None:

            if not isinstance(value, (list)):
                value = numpy.array(value)
            else:
                value = numpy.array([value])

            self.classifier.intercept_ = value

        # value is None
        elif value is None:
            try:
                del self.classifier.intercept_
            except AttributeError:
                pass
        else:
            raise ValueError('incorrect type')


class NaiveBayesClassifier(Classifier):
    """
    NaiveBayesClassifier(alpha=1.0)

    Naive Bayes Classifier.

    The `Naive Bayes classifier (wikipedia)
    <https://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_ partitions
    candidate record pairs into matches and non-matches. The classifier is
    based on probabilistic principles. The Naive Bayes classification method
    is proven to be mathematical equivalent with the Fellegi and Sunter model.

    Parameters
    ----------
    log_prior : list, numpy.array
        The log propabaility of each class.

    Attributes
    ----------
    classifier: sklearn.linear_model.LogisticRegression
        The Logistic regression classifier in sklearn.
    coefficients : list
        The coefficients of the logistic regression.
    intercept : float
        The interception value.

    Parameters
    ----------
    alpha : float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

    """

    def __init__(self, log_prior=None, alpha=1.0, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.classifier = naive_bayes.BernoulliNB(alpha=alpha, binarize=None)

        self.log_prior = log_prior

    @property
    def log_prior(self):

        try:
            return self.classifier.class_log_prior_.tolist()
        except Exception:
            return None

    @log_prior.setter
    def log_prior(self, value):

        if isinstance(value, (list, numpy.ndarray)):

            self.classifier.class_log_prior_ = numpy.array(value)

        # value is None
        elif value is None:
            try:
                del self.classifier.class_log_prior_
            except AttributeError:
                pass
        else:
            raise ValueError('incorrect type')


class SVMClassifier(Classifier):
    """
    SVMClassifier()

    Support Vector Machines

    The `Support Vector Machine classifier (wikipedia)
    <https://en.wikipedia.org/wiki/Support_vector_machine>`_ partitions
    candidate record pairs into matches and non-matches. This implementation
    is a non-probabilistic binary linear classifier. Support vector machines
    are supervised learning models. Therefore, the SVM classifiers needs
    training-data.

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.classifier = svm.LinearSVC()

    def prob(self, *args, **kwargs):

        raise AttributeError(
            "It is not possible to compute "
            "probabilities for the SVMClassfier")


class FellegiSunter(Classifier):
    """Fellegi and Sunter framework.

    Base class for probabilistic classification of records pairs with the
    Fellegi and Sunter (1969) framework.

    """

    def __init__(self, random_decision_rule=False):

        self.random_decision_rule = random_decision_rule

    @property
    def p(self):
        try:
            return self.algorithm._p
        except Exception:
            pass

    def _decision_rule(self, probabilities, threshold,
                       random_decision_rule=False):

        if not self.random_decision_rule:
            return (probabilities >= threshold).astype(int)
        else:
            raise NotImplementedError(
                "Random decisions are not possible at the moment.")


class ECMClassifier(FellegiSunter):
    """Expectation/Conditional Maxisation vlassifier.

    [EXPERIMENTAL] Expectation/Conditional Maximisation algorithm used as
    classifier. This probabilistic record linkage algorithm is used in
    combination with Fellegi and Sunter model.

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.algorithm = ECMEstimate()

    def learn(self, comparison_vectors, init='jaro', return_type='index'):
        """ Train the algorithm.

        Train the Expectation-Maximisation classifier. This method is well-
        known as the ECM-algorithm implementation in the context of record
        linkage.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The dataframe with comparison vectors.
        params_init : dict
            A dictionary with initial parameters of the ECM algorithm
            (optional).
        return_type : 'index' (default), 'series', 'array'
            The format to return the classification result. The argument value
            'index' will return the pandas.MultiIndex of the matches. The
            argument value 'series' will return a pandas.Series with zeros
            (distinct) and ones (matches). The argument value 'array' will
            return a numpy.ndarray with zeros and ones.

        Returns
        -------
        pandas.Series
            A pandas Series with the labels 1 (for the matches) and 0 (for the
            non-matches).

        """

        logging.info("Classifying - start learning {}".format(
            self.__class__.__name__)
        )

        # start timing
        start_time = time.time()

        probs = self.algorithm.train(comparison_vectors.as_matrix())

        n_matches = int(self.algorithm.p * len(probs))
        self.p_threshold = numpy.sort(probs)[len(probs) - n_matches]

        prediction = self._decision_rule(probs, self.p_threshold)

        result = self._return_result(
            prediction, return_type, comparison_vectors
        )

        # log timing
        logf_time = "Classifying - learning computation time: ~{:.2f}s"
        logging.info(logf_time.format(time.time() - start_time))

        return result

    def predict(self, comparison_vectors, return_type='index',
                *args, **kwargs):
        """Predict the class of reord pairs.

        Classify a set of record pairs based on their comparison vectors into
        matches, non-matches and possible matches. The classifier has to be
        trained to call this method.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The dataframe with comparison vectors.
        return_type : 'index' (default), 'series', 'array'
            The format to return the classification result. The argument value
            'index' will return the pandas.MultiIndex of the matches. The
            argument value 'series' will return a pandas.Series with zeros
            (distinct) and ones (matches). The argument value 'array' will
            return a numpy.ndarray with zeros and ones.

        Returns
        -------
        pandas.Series
            A pandas Series with the labels 1 (for the matches) and 0 (for the
            non-matches).

        Note
        ----
        Prediction is risky for this unsupervised learning method. Be aware
        that the sample from the population is valid.


        """

        logging.info("Classifying - predict matches and non-matches")

        enc_vectors = self.algorithm._transform_vectors(
            comparison_vectors.as_matrix())

        probs = self.algorithm._expectation(enc_vectors)

        prediction = self._decision_rule(probs, self.p_threshold)

        return self._return_result(prediction, return_type, comparison_vectors)

    def prob(self, comparison_vectors):
        """Compute the probabilities for each record pair.

        For each pair of records, estimate the probability of being a match.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The dataframe with comparison vectors.
        return_type : 'series' or 'array'
            Return a pandas series or numpy array. Default 'series'.

        Returns
        -------
        pandas.Series or numpy.ndarray
            The probability of being a match for each record pair.

        """

        logging.info("Classifying - compute probabilities")

        enc_vectors = self.algorithm._transform_vectors(
            comparison_vectors.as_matrix())

        return pandas.Series(
            self.algorithm._expectation(enc_vectors),
            index=comparison_vectors.index
        )
