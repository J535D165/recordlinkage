import numpy

from sklearn import cluster, linear_model, naive_bayes, svm

from recordlinkage.base import BaseClassifier as Classifier
from recordlinkage.base import SKLearnClassifier
from recordlinkage.algorithms.em_sklearn import ECM


class FellegiSunter(object):
    """Fellegi and Sunter (1969) framework.

    Meta class for probabilistic classification algorithms. The Fellegi and
    Sunter class is used for the :class:`recordlinkage.NaiveBayesClassifier`
    and :class:`recordlinkage.ECMClassifier`.

    Attributes
    ----------

    log_p : float
        Log match probability as described in the FS framework.

    log_m_probs : np.ndarray
        Log probability P(x_i==1|Match) as described in the FS framework.

    log_u_probs : np.ndarray
        Log probability P(x_i==1|Non-match) as described in the FS framework.

    log_weights : np.ndarray
        Log weights as described in the FS framework.

    p : float
        Match probability as described in the FS framework.

    m_probs : np.ndarray
        Probability P(x_i==1|Match) as described in the FS framework.

    u_probs : np.ndarray
        Probability P(x_i==1|Non-match) as described in the FS framework.

    weights : np.ndarray
        Weights as described in the FS framework.

    References
    ----------

    Fellegi, Ivan P and Alan B Sunter. 1969. “A theory for record linkage.”
    Journal of the American Statistical Association 64(328):1183–1210.

    """

    def _decision_rule(self, probabilities, threshold):

        return (probabilities >= threshold).astype(int)

    def _match_class_pos(self):
        # add notfitted warnings

        if self.kernel.classes_.shape[0] != 2:
            raise ValueError("Number of classes is {}, expected 2.".format(
                self.kernel.classes_.shape[0]))

        # get the position of match probabilities
        classes = list(self.kernel.classes_)
        return classes.index(1)

    def _nonmatch_class_pos(self):
        # add notfitted warnings

        if self.kernel.classes_.shape[0] != 2:
            raise ValueError("Number of classes is {}, expected 2.".format(
                self.kernel.classes_.shape[0]))

        # get the position of match probabilities
        classes = list(self.kernel.classes_)
        return classes.index(0)

    @property
    def log_p(self):
        """Log match probability as described in the FS framework."""
        return self.kernel.class_log_prior_[self._match_class_pos()]

    @log_p.setter
    def log_p(self, value):
        self.kernel.class_log_prior_[self._match_class_pos()] = value

    @property
    def log_m_probs(self):
        """Log probability P(x_i==1|Match) as described in the FS framework."""
        return self.kernel.feature_log_prob_[self._match_class_pos()]

    @log_m_probs.setter
    def log_m_probs(self, value):
        self.kernel.feature_log_prob_[self._match_class_pos()] = value

    @property
    def log_u_probs(self):
        """Log probability P(x_i==1|Non-match) as described in the FS framework.
        """
        return self.kernel.feature_log_prob_[self._nonmatch_class_pos()]

    @log_u_probs.setter
    def log_u_probs(self, value):
        self.kernel.feature_log_prob_[self._nonmatch_class_pos()] = value

    @property
    def log_weights(self):
        """Log weights as described in the FS framework."""

        return self.log_m_probs - self.log_u_probs

    @log_weights.setter
    def log_weights(self, value):
        raise AttributeError(
            "setting 'log_weights' or 'weights' is not possible"
        )

    @property
    def p(self):
        """Match probability as described in the FS framework."""

        return numpy.exp(self.log_p)

    @p.setter
    def p(self, value):
        self.__p = value

    @property
    def m_probs(self):
        """Probability P(x_i==1|Match) as described in the FS framework."""

        return numpy.exp(self.log_m_probs)

    @m_probs.setter
    def m_probs(self, value):
        self.__m_probs = numpy.log(value)

    @property
    def u_probs(self):
        """Probability P(x_i==1|Non-match) as described in the FS framework."""

        return numpy.exp(self.log_u_probs)

    @u_probs.setter
    def u_probs(self, value):
        self.__u_probs = numpy.log(value)

    @property
    def weights(self):
        """Weights as described in the FS framework."""

        return numpy.exp(self.log_weights)

    @weights.setter
    def weights(self, value):
        raise AttributeError(
            "setting 'log_weights' or 'weights' is not possible"
        )


class KMeansClassifier(SKLearnClassifier, Classifier):
    """KMeans classifier.

    The `K-means clusterings algorithm (wikipedia)
    <https://en.wikipedia.org/wiki/K-means_clustering>`_ partitions candidate
    record pairs into matches and non-matches. Each comparison vector belongs
    to the cluster with the nearest mean.

    The K-means algorithm is an unsupervised learning algorithm. The algorithm
    doesn't need trainings data for fitting. The algorithm is calibrated for
    two clusters: a match cluster and a non-match cluster). The centers of
    these clusters can be given as arguments or set automatically.

    The KMeansClassifier classifier uses the :class:`sklearn.cluster.KMeans`
    clustering algorithm from SciKit-learn as kernel.

    Parameters
    ----------

    match_cluster_center : list, numpy.array
        The center of the match cluster. The length of the list/array must
        equal the number of comparison variables. If None, the match cluster
        center is set automatically. Default None.

    nonmatch_cluster_center : list, numpy.array
        The center of the nonmatch (distinct) cluster. The length of the
        list/array must equal the number of comparison variables.  If None,
        the non-match cluster center is set automatically. Default None.

    **kwargs :
        Additional arguments to pass to :class:`sklearn.cluster.KMeans`.

    Attributes
    ----------

    kernel: sklearn.cluster.KMeans
        The kernel of the classifier. The kernel is
        :class:`sklearn.cluster.KMeans` from SciKit-learn.

    match_cluster_center : numpy.array
        The center of the match cluster.

    nonmatch_cluster_center : numpy.array
        The center of the nonmatch (distinct) cluster.

    Note
    ----

    There are better methods for linking records than the k-means clustering
    algorithm. This algorithm can be useful for an (unsupervised) initial
    partition.

    """

    def __init__(self,
                 match_cluster_center=None,
                 nonmatch_cluster_center=None,
                 **kwargs):
        super(KMeansClassifier, self).__init__()

        # initialize the classifier
        self.kernel = cluster.KMeans(n_clusters=2, n_init=1, **kwargs)

        # set cluster centers if available
        self.match_cluster_center = match_cluster_center
        self.nonmatch_cluster_center = nonmatch_cluster_center

    def _initialise_classifier(self, comparison_vectors):
        """Set the centers of the clusters."""

        # Set the start point of the classifier.
        self.kernel.init = numpy.array(
            [[0.05] * len(list(comparison_vectors)),
             [0.95] * len(list(comparison_vectors))])

    @property
    def match_cluster_center(self):
        return self.kernel.cluster_centers_[1, :]

    @match_cluster_center.setter
    def match_cluster_center(self, value):

        if value is None:
            return

        if not hasattr(self.kernel, 'cluster_centers_'):
            self.kernel.cluster_centers_ = numpy.empty((2, len(value)))
            self.kernel.cluster_centers_[:] = numpy.nan

        self.kernel.cluster_centers_[1, :] = numpy.asarray(value)

    @property
    def nonmatch_cluster_center(self):
        return self.kernel.cluster_centers_[0, :]

    @nonmatch_cluster_center.setter
    def nonmatch_cluster_center(self, value):

        if value is None:
            return

        if not hasattr(self.kernel, 'cluster_centers_'):
            self.kernel.cluster_centers_ = numpy.empty((2, len(value)))
            self.kernel.cluster_centers_[:] = numpy.nan

        self.kernel.cluster_centers_[0, :] = numpy.asarray(value)

    def prob(self, *args, **kwargs):

        raise AttributeError("It is not possible to compute "
                             "probabilities for the KMeansClassfier")


class LogisticRegressionClassifier(SKLearnClassifier, Classifier):
    """Logistic Regression Classifier.

    This classifier is an application of the `logistic regression model
    (wikipedia) <https://en.wikipedia.org/wiki/Logistic_regression>`_. The
    classifier partitions candidate record pairs into matches and non-matches.

    This algorithm is also known as Deterministic Record Linkage.

    The LogisticRegressionClassifier classifier uses the
    :class:`sklearn.linear_model.LogisticRegression` classification algorithm
    from SciKit-learn as kernel.

    Parameters
    ----------

    coefficients : list, numpy.array
        The coefficients of the logistic regression.

    intercept : float
        The interception value.

    **kwargs :
        Additional arguments to pass to
        :class:`sklearn.linear_model.LogisticRegression`.

    Attributes
    ----------

    kernel: sklearn.linear_model.LogisticRegression
        The kernel of the classifier. The kernel is
        :class:`sklearn.linear_model.LogisticRegression` from SciKit-learn.

    coefficients : list
        The coefficients of the logistic regression.

    intercept : float
        The interception value.

    """

    def __init__(self,
                 coefficients=None,
                 intercept=None,
                 **kwargs):
        super(LogisticRegressionClassifier, self).__init__()

        self.coefficients = coefficients
        self.intercept = intercept

        self.kernel = linear_model.LogisticRegression(**kwargs)
        self.kernel.classes_ = numpy.array([0, 1])

    @property
    def params(self):
        return {'coefficients': self.coefficients, 'intercept': self.intercept}

    @params.setter
    def params(self, value):

        if not isinstance(value, dict):
            raise ValueError("parameters are of wrong type")

        self.coefficients = value['coefficients']
        self.intercept = value['intercept']

    @property
    def coefficients(self):
        return self.kernel.coef_[0]

    @coefficients.setter
    def coefficients(self, value):

        if value is not None:
            value = numpy.asarray(value)
            self.kernel.coef_ = value.reshape((1, len(value)))
        else:
            try:
                del self.kernel.coef_
            except AttributeError:
                pass

    @property
    def intercept(self):
        return self.kernel.intercept_[0]

    @intercept.setter
    def intercept(self, value):

        if value is not None:
            value = numpy.asarray(value)
            value = numpy.atleast_2d(value)
            self.kernel.intercept_ = value
        else:
            try:
                del self.kernel.intercept_
            except AttributeError:
                pass


class NaiveBayesClassifier(SKLearnClassifier, Classifier, FellegiSunter):
    """Naive Bayes Classifier.

    The `Naive Bayes classifier (wikipedia)
    <https://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_ partitions
    candidate record pairs into matches and non-matches. The classifier is
    based on probabilistic principles. The Naive Bayes classification method
    is proven to be mathematical equivalent with the Fellegi and Sunter model.

    The NaiveBayesClassifier classifier uses the
    :class:`sklearn.naive_bayes.BernoulliNB` classification algorithm from
    SciKit-learn as kernel.

    Parameters
    ----------

    alpha : float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        Default 0.0.

    **kwargs :
        Additional arguments to pass to
        :class:`sklearn.naive_bayes.BernoulliNB`.

    Attributes
    ----------

    kernel: sklearn.naive_bayes.BernoulliNB
        The kernel of the classifier. The kernel is
        :class:`sklearn.naive_bayes.BernoulliNB` from SciKit-learn.

    log_p : float
        Log match probability as described in the FS framework.

    log_m_probs : np.ndarray
        Log probability P(x_i==1|Match) as described in the FS framework.

    log_u_probs : np.ndarray
        Log probability P(x_i==1|Non-match) as described in the FS framework.

    log_weights : np.ndarray
        Log weights as described in the FS framework.

    p : float
        Match probability as described in the FS framework.

    m_probs : np.ndarray
        Probability P(x_i==1|Match) as described in the FS framework.

    u_probs : np.ndarray
        Probability P(x_i==1|Non-match) as described in the FS framework.

    weights : np.ndarray
        Weights as described in the FS framework.

    """

    def __init__(self,
                 alpha=0.0,
                 **kwargs):
        super(NaiveBayesClassifier, self).__init__()

        self.kernel = naive_bayes.BernoulliNB(
            alpha=alpha,
            binarize=None,
            **kwargs
        )


class SVMClassifier(SKLearnClassifier, Classifier):
    """Support Vector Machines Classifier

    The `Support Vector Machine classifier (wikipedia)
    <https://en.wikipedia.org/wiki/Support_vector_machine>`_ partitions
    candidate record pairs into matches and non-matches. This implementation
    is a non-probabilistic binary linear classifier. Support vector machines
    are supervised learning models. Therefore, SVM classifiers need training-
    data.

    The SVMClassifier classifier uses the :class:`sklearn.svm.LinearSVC`
    classification algorithm from SciKit-learn as kernel.

    Parameters
    ----------

    **kwargs :
        Arguments to pass to :class:`sklearn.svm.LinearSVC`.

    Attributes
    ----------

    kernel: sklearn.svm.LinearSVC
        The kernel of the classifier. The kernel is
        :class:`sklearn.svm.LinearSVC` from SciKit-learn.

    """

    def __init__(self, *args, **kwargs):
        super(SVMClassifier, self).__init__()

        self.kernel = svm.LinearSVC(*args, **kwargs)

    def prob(self, *args, **kwargs):

        raise AttributeError("It is not possible to compute "
                             "probabilities for the SVMClassfier")


class ECMClassifier(SKLearnClassifier, Classifier, FellegiSunter):
    """Expectation/Conditional Maxisation classifier (Unsupervised).

    Expectation/Conditional Maximisation algorithm used as
    classifier. This probabilistic record linkage algorithm is used in
    combination with Fellegi and Sunter model.

    Parameters
    ----------

    binarize : float or None, optional (default=0.0)
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    max_iter : int
        The maximum number of iterations of the EM algorithm. Default 100.

    atol : float
        The tolerance between parameters between each interation. If the
        difference between the parameters between the iterations is smaller
        than this value, the algorithm is considered to be converged.
        Default 10e-4.

    Attributes
    ----------

    kernel: recordlinkage.algorithms.em_sklearn.ECM
        The kernel of the classifier.

    log_p : float
        Log match probability as described in the FS framework.

    log_m_probs : np.ndarray
        Log probability P(x_i==1|Match) as described in the FS framework.

    log_u_probs : np.ndarray
        Log probability P(x_i==1|Non-match) as described in the FS framework.

    log_weights : np.ndarray
        Log weights as described in the FS framework.

    p : float
        Match probability as described in the FS framework.

    m_probs : np.ndarray
        Probability P(x_i==1|Match) as described in the FS framework.

    u_probs : np.ndarray
        Probability P(x_i==1|Non-match) as described in the FS framework.

    weights : np.ndarray
        Weights as described in the FS framework.

    References
    ----------

    Herzog, Thomas N, Fritz J Scheuren and William E Winkler. 2007. Data
    quality and record linkage techniques. Vol. 1 Springer.

    """

    def __init__(self, *args, **kwargs):
        super(ECMClassifier, self).__init__()

        self.kernel = ECM(*args, **kwargs)

    @property
    def algorithm(self):
        # Deprecated
        raise AttributeError(
            "This attribute is deprecated. Use 'classifier' instead.")
