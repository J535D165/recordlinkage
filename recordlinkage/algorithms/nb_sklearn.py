"""

This module implements the Naive Bayes algorithm in both a supervised and
unsupervised way. These are unsupervised learning methods based on applying
Bayes' theorem with strong (naive) feature independence assumptions. The
module is based on scikit-learn's subdule :mod:`sklearn.naive_bayes`.

"""

# This module is based on sklearn's NB implementation. The license of sklearn
# is BSD 3 clause. Modifications copyright Jonathan de Bruin.

import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import binarize
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y

try:  # SciPy >= 0.19
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp  # noqa

from scipy.sparse import issparse

from recordlinkage.types import is_string_like

__all__ = ["NaiveBayes", "ECM"]

_ALPHA_MIN = 1e-10


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this method."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.
    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def unique_rows_counts(a):
    lidx = np.ravel_multi_index(a.T, a.max(0) + 1)
    _, unq_idx, counts = np.unique(lidx, return_index=True, return_counts=True)
    return a[unq_idx], counts


class BaseNB(BaseEstimator, ClassifierMixin):
    """Abstract base class for naive Bayes estimators"""

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse="csr")
        X_bin = self._transform_data(X)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X_bin.shape

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (n_features, n_features_X)
            )

        # see chapter 4.1 of http://www.cs.columbia.edu/~mcollins/em.pdf
        # implementation as in Formula 4.
        jll = safe_sparse_dot(X_bin, self.feature_log_prob_.T)
        jll += self.class_log_prior_

        return jll

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        jll = self._joint_log_likelihood(X)

        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)  # return shape = (2,)

        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        return np.exp(self.predict_log_proba(X))

    def _fit_data(self, X):
        """Binarize the data for each column separately.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_transformed : array-like
            Returns the data where in each columns the labels are
            binarized.

        """

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        for i in range(X.shape[1]):
            # initialise binarizer and save
            binarizer = LabelBinarizer()

            if self.binarize:
                binarizer.classes_ = np.array([0, 1])

            # fit the data to the binarizer
            binarizer.fit(X[:, i])

            self._binarizers.append(binarizer)

        return self._transform_data(X)

    def _transform_data(self, X):
        """Binarize the data for each column separately."""

        if self._binarizers == []:
            raise NotFittedError()

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        if len(self._binarizers) != X.shape[1]:
            raise ValueError(
                "Expected input with %d features, got %d instead"
                % (len(self._binarizers), X.shape[1])
            )

        X_parts = []

        for i in range(X.shape[1]):
            X_i = self._binarizers[i].transform(X[:, i])

            # sklearn returns ndarray with shape (samples, 1) on binary input.
            if self._binarizers[i].classes_.shape[0] == 1:
                X_parts.append(1 - X_i)
            elif self._binarizers[i].classes_.shape[0] == 2:
                X_parts.append(1 - X_i)
                X_parts.append(X_i)
            else:
                X_parts.append(X_i)

        return np.concatenate(X_parts, axis=1)

    @property
    def labels_(self):
        c = []
        for _i, bin in enumerate(self._binarizers):
            c.append(list(bin.classes_))

        return c


class NaiveBayes(BaseNB):
    """NaiveBayes classifier for multivariate models.

    Parameters
    ----------

    binarize : float or None, optional (default=0.0)
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    alpha: float
        Default 1.0.

    fit_prior: bool
        Default True.

    class_prior: np.array
        Default None.

    Attributes
    ----------
    class_log_prior_ : array, shape = [n_classes]
        Log probability of each class (smoothed).

    feature_log_prob_ : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    """

    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_log_prior_ = class_prior
        self.class_prior = class_prior

        self._binarizers = []

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""

        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_ + alpha * 2

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of" " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = np.log(self.class_count_) - np.log(
                self.class_count_.sum()
            )
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError(
                "Smoothing parameter alpha = %.1e. "
                "alpha should be > 0." % np.min(self.alpha)
            )
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.feature_count_.shape[1]:
                raise ValueError(
                    "alpha should be a scalar or a numpy array "
                    "with shape [n_features]"
                )
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn(
                "alpha too small will result in numeric errors, "
                "setting alpha = %.1e" % _ALPHA_MIN,
                stacklevel=2,
            )
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, "csr")

        # Transform data with a label binarizer. Each column will get
        # transformed into a N columns (for each distinct value a column). For
        # a situation with 0 and 1 outcome values, the result given two
        # columns.
        X_bin = self._fit_data(X)
        _, n_features = X_bin.shape

        # prepare Y
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros(
            (n_effective_classes, n_features), dtype=np.float64
        )
        self._count(X_bin, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)

        return self

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (
            self.feature_log_prob_[1:]
            if len(self.classes_) == 2
            else self.feature_log_prob_
        )

    def _get_intercept(self):
        return (
            self.class_log_prior_[1:]
            if len(self.classes_) == 2
            else self.class_log_prior_
        )

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)


class ECM(BaseNB):
    """ECM classifier for multivariate Bernoulli models.

    ECM is designed for binary/boolean features.

    Parameters
    ----------

    init : str
        Initialisation method for the algorithm. Options are:
        'jaro' and 'random'. Default 'jaro'.
    max_iter : int
        Maximum number of iterations. Default 100.
    binarize : float or None, optional (default=0.0)
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    atol : float
        Difference between trainable parameters between iterations
        needs to be less than this value. If less than this value, the
        algorithm is considered to be converged. Default 10e-4.


    Attributes
    ----------
    class_log_prior_ : array, shape = [n_classes]
        Log probability of each class (smoothed).

    feature_log_prob_ : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    classes_ : array, shape = [2,]
        The outcome classes (match, non-match). Default [0, 1]
    """

    # TODO: set the init parameters by the user

    def __init__(self, init="jaro", max_iter=100, binarize=binarize, atol=10e-5):
        self.init = init
        self.max_iter = max_iter
        self.binarize = binarize
        self.atol = atol

        self._binarizers = []

    def _init_parameters_random(self, X_bin):
        """Initialise parameters for unsupervised learning."""

        _, n_features = X_bin.shape

        # The parameter class_log_prior_ has shape (2,). The values represent
        # 'match' and 'non-match'.
        rand_vals = np.random.rand(2)
        class_prior = rand_vals / np.sum(rand_vals)

        # make empty array of feature log probs
        # dimensions 2xn_features
        feature_prob = np.zeros((2, n_features))

        feat_i = 0

        for _i, bin in enumerate(self._binarizers):
            bin_len = bin.classes_.shape[0]

            rand_vals_0 = np.random.rand(bin_len)
            feature_prob[0, feat_i : feat_i + bin_len] = rand_vals_0 / np.sum(
                rand_vals_0
            )

            rand_vals_1 = np.random.rand(bin_len)
            feature_prob[1, feat_i : feat_i + bin_len] = rand_vals_1 / np.sum(
                rand_vals_1
            )

            feat_i += bin_len

        return np.log(class_prior), np.log(feature_prob)

    def _init_parameters_jaro(self, X_bin):
        """Initialise parameters for unsupervised learning."""

        _, n_features = X_bin.shape

        class_prior = [0.9, 0.1]
        feature_prob = []

        for i, bin in enumerate(self._binarizers):
            if bin.classes_.shape[0] > 2:
                raise ValueError(
                    "Only binary labels are allowed for "
                    "'jaro'method. "
                    "Column {} has {} different labels.".format(
                        i, bin.classes_.shape[0]
                    )
                )

            for binclass in bin.classes_:
                if binclass == 1:
                    feature_prob.append([0.1, 0.9])
                if binclass == 0:
                    feature_prob.append([0.9, 0.1])

        return np.log(class_prior), np.log(feature_prob).T

    def fit(self, X):
        """Fit ECM classifier according to X

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse="csr")

        # count frequencies of elements in vector space
        # based on https://stackoverflow.com/a/33235665
        # faster than numpy.unique
        X_unique, X_freq = np.unique(X, axis=0, return_counts=True)
        X_freq = np.atleast_2d(X_freq)

        # Transform data with a label binarizer. Each column will get
        # transformed into a N columns (for each distinct value a column). For
        # a situation with 0 and 1 outcome values, the result given two
        # columns.
        X_unique_bin = self._fit_data(X_unique)
        _, n_features = X_unique_bin.shape

        # initialise parameters
        self.classes_ = np.array([0, 1])

        if is_string_like(self.init) and self.init == "random":
            (
                self.class_log_prior_,
                self.feature_log_prob_,
            ) = self._init_parameters_random(X_unique_bin)
        elif is_string_like(self.init) and self.init == "jaro":
            self.class_log_prior_, self.feature_log_prob_ = self._init_parameters_jaro(
                X_unique_bin
            )
        else:
            raise ValueError(
                "'{}' is not a valid value for " "argument 'init'".format(self.init)
            )

        iteration = 0
        stop_iteration = False

        self._logging_class_log_prior = np.atleast_2d(self.class_log_prior_)
        self._logging_feature_log_prob = np.atleast_3d(self.feature_log_prob_)

        while iteration < self.max_iter and not stop_iteration:
            # Increment counter
            iteration += 1

            # expectation step
            g = self.predict_proba(X_unique)
            g_freq = g * X_freq.T
            g_freq_sum = g_freq.sum(axis=0)

            # maximisation step
            class_log_prior_ = np.log(g_freq_sum) - np.log(X.shape[0])  # p
            feature_prob_ = safe_sparse_dot(g_freq.T, X_unique_bin)
            feature_log_prob_ = np.log(feature_prob_)
            feature_log_prob_ -= np.log(np.atleast_2d(g_freq_sum).T)

            # Stop iterating when the class prior and feature probs are close
            # to the values in the to previous iteration (parameters starting
            # with 'self').
            if self.atol is not None:
                class_log_prior_close = np.allclose(
                    np.exp(class_log_prior_),
                    np.exp(self.class_log_prior_),
                    atol=self.atol,
                )
                feature_log_prob_close = np.allclose(
                    np.exp(feature_log_prob_),
                    np.exp(self.feature_log_prob_),
                    atol=self.atol,
                )

                if class_log_prior_close and feature_log_prob_close:
                    stop_iteration = True
                    logging.info(
                        f"ECM algorithm converged after {iteration} iterations"
                    )

            if np.all(np.isnan(feature_log_prob_)):
                logging.warning(
                    "ECM algorithm might not converged correctly after "
                    "{} iterations".format(iteration),
                    stacklevel=2,
                )
                break

            # Update the class prior and feature probs.
            self.class_log_prior_ = class_log_prior_
            self.feature_log_prob_ = feature_log_prob_

            # create logs
            self._logging_class_log_prior = np.concatenate(
                [self._logging_class_log_prior, np.atleast_2d(self.class_log_prior_)]
            )
            self._logging_feature_log_prob = np.concatenate(
                [self._logging_feature_log_prob, np.atleast_3d(self.feature_log_prob_)],
                axis=2,
            )
        else:
            if iteration == self.max_iter:
                logging.info(
                    f"ECM algorithm stopped at {iteration} (=max_iter) iterations"
                )

        return self

    # The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (
            self.feature_log_prob_[1:]
            if len(self.classes_) == 2
            else self.feature_log_prob_
        )

    def _get_intercept(self):
        return (
            self.class_log_prior_[1:]
            if len(self.classes_) == 2
            else self.class_log_prior_
        )

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)
