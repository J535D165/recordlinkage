# -*- coding: utf-8 -*-
"""

This module implements ECM algorithms. These are unsupervised learning methods
based on applying Bayes' theorem with strong (naive) feature independence
assumptions. The module is based on scikit-learn's subdule
:mod:`sklearn.naive_bayes`.


"""

# This module is based on sklearn's NB implementation. The license of sklearn
# is BSD 3 clause. Modifications copyright Jonathan de Bruin.


from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import binarize
from sklearn.utils.validation import check_array
from sklearn.exceptions import NotFittedError

try:  # SciPy >= 0.19
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp  # noqa
from scipy.sparse import issparse

import six

__all__ = ['ECM']


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
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


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


class BaseNB(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape [n_classes, n_samples].

        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        """

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
        log_prob_x = logsumexp(jll, axis=1)
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


class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

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
        X = check_array(X, accept_sparse='csr')
        _, n_features = X.shape

        binarize_value = self.binarize

        # initialise parameters
        self.classes_ = np.array([0, 1])
        self.class_log_prior_ = np.log(np.array([.5, .5]))
        feature_log_prob = np.zeros((2, n_features))
        feature_log_prob[0, :] = 0.1
        feature_log_prob[1, :] = 0.9
        self.feature_log_prob_ = np.log(feature_log_prob)

        # Binarize values and set binarize to None temporary. This is done to
        # speed up iteration process.
        if binarize_value is not None:
            X = binarize(X, threshold=binarize_value)
        self.binarize = None

        # count frequencies
        # based on https://stackoverflow.com/a/33235665
        # faster than numpy.unique
        X_unique, X_freq = np.unique(X, axis=0, return_counts=True)
        X_freq = np.atleast_2d(X_freq)

        iteration = 0
        stop_iteration = False

        while iteration < self.max_iter and not stop_iteration:

            # expectation step
            g = self.predict_proba(X_unique)
            g_freq = g * X_freq.T
            g_freq_sum = g_freq.sum(axis=0)

            # maximisation step
            class_log_prior_ = np.log(g_freq_sum) - np.log(X.shape[0])  # p
            feature_log_prob_ = np.log(safe_sparse_dot(g_freq.T, X_unique))
            feature_log_prob_ -= np.log(np.atleast_2d(g_freq_sum).T)

            # Stop iterating when parameters are close to previous iteration
            class_log_prior_close = np.allclose(
                class_log_prior_, self.class_log_prior_, atol=self.atol)
            feature_log_prob_close = np.allclose(
                feature_log_prob_, self.feature_log_prob_, atol=self.atol)
            if (class_log_prior_close and feature_log_prob_close):
                stop_iteration = True
            if np.all(np.isnan(feature_log_prob_)):
                stop_iteration = True

            self.class_log_prior_ = class_log_prior_
            self.feature_log_prob_ = feature_log_prob_

            # Increment counter
            iteration += 1

        self.binarize = binarize_value

        return self

    # XXX The following is a stopgap measure; we need to set the dimensions
    # of class_log_prior_ and feature_log_prob_ correctly.
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)

    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)

    coef_ = property(_get_coef)
    intercept_ = property(_get_intercept)


class ECM(BaseDiscreteNB):
    """ECM classifier for multivariate Bernoulli models.

    ECM is designed for binary/boolean features.

    Parameters
    ----------

    binarize : float or None, optional (default=0.0)
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.

    Attributes
    ----------
    class_log_prior_ : array, shape = [n_classes]
        Log probability of each class (smoothed).

    feature_log_prob_ : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    """

    def __init__(self,
                 binarize=.8,
                 max_iter=100,
                 atol=10e-4):
        self.binarize = binarize
        self.max_iter = max_iter
        self.atol = atol

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError(
                "Expected input with %d features, got %d instead" %
                (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X, (self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)

        return jll
