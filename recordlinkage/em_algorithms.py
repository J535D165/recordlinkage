
# futures
from __future__ import division

from itertools import groupby

# external
import numpy

from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class EMEstimate(object):
    pass


class ECMEstimate(EMEstimate):
    """

    Algorithm to compute the Expectation/Conditional Maximisation algorithm in
    the context of record linkage. The algorithm is clearly described by
    Herzog, Schueren and Winkler in the book: Data Quality and Record Linkage
    Tehniques. The algorithm assumes that the comparison variables are
    mutually independent given the match status.

    :param max_iter: An integer specifying the maximum number of
                    iterations. Default maximum number of iterations is 100.
                    If max_iter=-1, there is no maximum number of iterations.

    :type max_iter: int


    """

    def __init__(self, max_iter=100, init='jaro', m=None, u=None, p=None):
        super(ECMEstimate, self).__init__()

        self.max_iter = max_iter
        self.init = init
        self.m = m
        self.u = u
        self.p = p

    def train(self, vectors):
        """

        Start the estimation of parameters with the iterative ECM-algorithm.

        """

        # If not np.ndarray, convert it
        vectors = numpy.array(vectors)

        # One hot encoding
        y = self._fit_transform_vectors(vectors)

        # Convert m and u parameters into lists
        # If init is list of numpy.array
        if isinstance(self.init, (list, numpy.ndarray)):
            try:
                self._m = numpy.array(
                    [self.m[cl][f]
                     for cl, f in zip(self._classes, self._features)])
                self._u = numpy.array(
                    [self.u[cl][f]
                     for cl, f in zip(self._classes, self._features)])
                self._p = self.p
            except Exception:
                raise ValueError("The parameters m and/or u are not correct. ")

        # If init is 'jaro'
        elif self.init in ['jaro', 'auto']:

            if numpy.all(numpy.in1d(self._features, [0, 1])):
                raise ValueError(
                    "To use 'jaro' for start point estimation, " +
                    "the feature values must be valued 1 or 0. ")

            self._m = 0.1 + 0.8 * self._classes
            self._u = 0.9 - 0.8 * self._classes
            self._p = 0.1
        else:
            raise ValueError("Method not known")

        self._iteration = 0

        # Iterate until converged
        while self._iteration < self.max_iter or self.max_iter == -1:

            prev_m, prev_u, prev_p = self._m, self._u, self._p

            # Expectation step
            g = self._expectation(y)

            # Maximisation step
            self._m, self._u, self._p = self._maximization(y, g)

            # Increment counter
            self._iteration += 1

            # Stop iterating when parameters are close to previous iteration
            if (numpy.allclose(prev_m, self._m, atol=10e-5) and
                numpy.allclose(prev_u, self._u, atol=10e-5) and
                    numpy.allclose(prev_p, self._p, atol=10e-5)):
                break

        # Store the values under
        self.m = [{t1: t2 for _, t1, t2 in group} for key, group in groupby(
            zip(self._features, self._classes, self._m), lambda x: x[0])]
        self.u = [{t1: t2 for _, t1, t2 in group} for key, group in groupby(
            zip(self._features, self._classes, self._u), lambda x: x[0])]
        self.p = self._p

        return g

    def _maximization(self, y_enc, g):
        """

        Maximisation step of the ECM-algorithm.

        :param samples: Dataframe with comparison vectors.
        :param weights: The number of times the comparison vectors
                        samples occur. This frame needs to have the
                        same index as samples.
        :param prob: The expectation of comparison vector in samples.

        :return: A dict of marginal m-probabilities, a dict of marginal
                        u-probabilities and the match prevalence.
        :rtype: (dict, dict, float)

        """

        m = g.T * y_enc / numpy.sum(g)
        u = (1 - g).T * y_enc / numpy.sum(1 - g)
        p = numpy.average(g)

        return m, u, p

    def _expectation(self, y_enc):
        """

        Compute the expectation of the given comparison vectors.

        :return: A Series with the expectation.
        :rtype: pandas.Series
        """

        # The following approach has a lot of computational advantages. But if
        # there is a better method, replace it. See Herzog, Scheuren and
        # Winkler for details about the algorithm.
        m = numpy.exp(y_enc.dot(numpy.log(self._m)))
        u = numpy.exp(y_enc.dot(numpy.log(self._u)))
        p = self._p

        return p * m / (p * m + (1 - p) * u)

    def _fit_transform_vectors(self, vectors):
        """

        Encode the feature vectors with one-hot-encoding. ONLY FOR INTERNAL
        USE.

        :param vectors: The feature vectors
        :type vectors: numpy.ndarray

        :return: Sparse matrix with encoded features.
        :rtype: scipy.coo_matrix

        """

        n_samples, n_features = vectors.shape

        data_enc = []

        self._features = numpy.array([])  # Feature names
        self._classes = numpy.array([])  # Feature values

        self._label_encoders = [LabelEncoder() for i in range(0, n_features)]
        self._one_hot_encoders = [OneHotEncoder()
                                  for i in range(0, n_features)]

        for i in range(0, n_features):

            # scikit learn encoding
            label_encoded = self._label_encoders[
                i].fit_transform(vectors[:, i]).reshape((-1, 1))
            data_enc_i = self._one_hot_encoders[i].fit_transform(label_encoded)

            # Save the classes and features in numpy arrays
            self._features = numpy.append(self._features, numpy.repeat(
                i, len(self._label_encoders[i].classes_)))  # Feature names
            self._classes = numpy.append(self._classes, self._label_encoders[
                                         i].classes_)  # Feature values

            # Append the encoded data to the dataframe
            data_enc.append(data_enc_i)

        return hstack(data_enc)

    def _transform_vectors(self, vectors):
        """

        Encode the feature vectors with one-hot-encoding.

        :param vectors: The feature vectors
        :type vectors: numpy.ndarray

        :return: Sparse matrix with encoded features.
        :rtype: scipy.coo_matrix

        """

        data_enc = []

        for i in range(0, len(self._label_encoders)):

            # scikit learn encoding
            label_encoded = self._label_encoders[
                i].transform(vectors[:, i]).reshape((-1, 1))
            data_enc_i = self._one_hot_encoders[i].transform(label_encoded)

            # Append the encoded data to the dataframe
            data_enc.append(data_enc_i)

        return hstack(data_enc)
