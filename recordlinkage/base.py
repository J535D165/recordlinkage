# -*- coding: utf-8 -*-

"""Base module for record linkage."""

from __future__ import division

import time
import warnings
from abc import ABCMeta, abstractmethod

from joblib import Parallel, delayed, cpu_count

import numpy as np

import pandas

from recordlinkage import rl_logging as logging
import recordlinkage.config as cf
from recordlinkage.utils import (listify,
                                 unique,
                                 is_label_dataframe,
                                 return_type_deprecator,
                                 index_split,
                                 frame_indexing)
from recordlinkage.types import (is_numpy_like,
                                 is_list_like,
                                 is_pandas_2d_multiindex)
from recordlinkage.measures import max_pairs
from recordlinkage.utils import DeprecationHelper, LearningError


def _parallel_compare_helper(class_obj, pairs, x, x_link=None):
    """Internal function to overcome pickling problem in python2."""
    return class_obj._compute(pairs, x, x_link)


def chunk_pandas(frame_or_series, chunksize=None):
    """Chunk a frame into smaller, equal parts."""
    if not isinstance(chunksize, int):
        raise ValueError('argument chunksize needs to be integer type')

    bins = np.arange(0, len(frame_or_series), step=chunksize)

    for b in bins:

        yield frame_or_series[b:b + chunksize]


class BaseIndex(object):
    """Base class for all index classes in Python Record Linkage Toolkit.

    Can be used for index passes.

    """

    def __init__(self, algorithms=[]):

        logging.info("indexing - initialize {} class".format(
            self.__class__.__name__)
        )

        self.algorithms = []
        self.add(algorithms)

        # logging
        self._i = 1
        self._i_max = None
        self._n = []
        self._n_max = []
        self._eta = []
        self._output_log_total = True

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{}>".format(class_name)

    def __str__(self):
        return repr(self)

    def add(self, model):
        """Add a index method.

        This method is used to add index algorithms. If multiple algorithms
        are added, the union of the record pairs from the algorithm is taken.

        Parameters
        ----------
        model : list, class
            A (list of) index algorithm(s) from
            :mod:`recordlinkage.index`.
        """
        if isinstance(model, list):
            self.algorithms = self.algorithms + model
        else:
            self.algorithms.append(model)

    def index(self, x, x_link=None):
        """Make an index of record pairs.

        Parameters
        ----------
        x: pandas.DataFrame
            A pandas DataFrame. When `x_link` is None, the algorithm makes
            record pairs within the DataFrame. When `x_link` is not empty,
            the algorithm makes pairs between `x` and `x_link`.
        x_link: pandas.DataFrame, optional
            A second DataFrame to link with the DataFrame x.

        Returns
        -------
        pandas.MultiIndex
            A pandas.MultiIndex with record pairs. Each record pair contains
            the index labels of two records.

        """
        if not self.algorithms:
            raise ValueError("No algorithms given.")

        # start timing
        start_time = time.time()

        pairs = None
        for cl_alg in self.algorithms:
            pairs_i = cl_alg.index(x, x_link)

            if pairs is None:
                pairs = pairs_i
            else:
                pairs = pairs.union(pairs_i)

        if x_link is not None:
            n_max = max_pairs((x, x_link))
        else:
            n_max = max_pairs(x)

        # store the number of pairs
        n = pairs.shape[0]
        eta = time.time() - start_time
        rr = 1 - n / n_max
        i_max = '?' if self._i_max is None else self._i_max

        self._eta.append(eta)
        self._n.append(n)
        self._n_max.append(n_max)

        # log
        logging.info("indexing [{:d}/{}] - time: {:.2f}s - pairs: {:d}/{:d} - "
                     "rr: {:0.5f}".format(self._i, i_max, eta, n, n_max, rr))

        # log total
        if self._output_log_total:

            n_total = np.sum(self._n)
            n_max_total = np.sum(self._n_max)
            rr_avg = 1 - n_total / n_max_total
            eta_total = np.sum(self._eta)

            logging.info("indexing [{:d}/{}] - time: {:.2f}s - "
                         "pairs_total: {:d}/{:d} - rr_total: {:0.5f}".format(
                             self._i, i_max, eta_total,
                             n_total, n_max_total, rr_avg))

        self._i += 1

        return pairs


class BaseIndexAlgorithm(object):
    """Base class for all index algorithms.

    BaseIndexAlgorithm is an abstract class for indexing algorithms.
    The method
    :func:`~recordlinkage.base.BaseIndexAlgorithm._link_index`


    Parameters
    ----------
    verify_integrity : bool
        Verify the integrity of the input dataframe(s). The index is
        checked for duplicate values.
    suffixes : tuple
        If the names of the resulting MultiIndex are identical, the
        suffixes are used to distinguish the names.

    Example
    -------
    Make your own indexation class::

        class CustomIndex(BaseIndexAlgorithm):

            def _link_index(self, df_a, df_b):

                # Custom link index.

                return ...

            def _dedup_index(self, df_a):

                # Custom deduplication index, optional.

                return ...

    Call the class in the same way::

        custom_index = CustomIndex():
        custom_index.index()

    """
    name = None
    description = None

    def __init__(self, verify_integrity=True, suffixes=('_1', '_2')):
        super(BaseIndexAlgorithm, self).__init__()

        self.suffixes = suffixes
        self.verify_integrity = verify_integrity

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{}>".format(class_name)

    def __str__(self):
        return repr(self)

    def _deduplication(self, x):

        if isinstance(x, (tuple, list)) and len(x) > 1:
            return False
        else:
            return True

    def _verify_integrety(self, x):

        if isinstance(x.index, pandas.Index):

            if not x.index.is_unique:
                raise ValueError('index of DataFrame is not unique')

        elif isinstance(x.index, pandas.MultiIndex):
            raise ValueError(
                'expected pandas.Index instead of pandas.MultiIndex'
            )

    def _link_index(self, df_a, df_b):
        """Build an index for linking two datasets.

        Parameters
        ----------
        df_a : (tuple of) pandas.Series
            The data of the left DataFrame to build the index with.
        df_b : (tuple of) pandas.Series
            The data of the right DataFrame to build the index with.

        Returns
        -------
        pandas.MultiIndex
            A pandas.MultiIndex with record pairs. Each record pair
            contains the index values of two records.

        """
        raise NotImplementedError(
            "Not possible to call index for the BaseEstimator"
        )

    def _dedup_index(self, df_a):
        """Build an index for deduplicating a dataset.

        Parameters
        ----------
        df_a : (tuple of) pandas.Series
            The data of the DataFrame to build the index with.

        Returns
        -------
        pandas.MultiIndex
            A pandas.MultiIndex with record pairs. Each record pair
            contains the index values of two records. The records are
            sampled from the lower triangular part of the matrix.
        """
        pairs = self._link_index(df_a, df_a)

        # Remove all pairs not in the lower triangular part of the matrix.
        # This part can be inproved by not comparing the level values, but the
        # level itself.
        try:
            pairs = pairs[pairs.codes[0] > pairs.codes[1]]
        except AttributeError:
            # backwards compat pandas <24
            pairs = pairs[pairs.labels[0] > pairs.labels[1]]

        return pairs

    def _make_index_names(self, name1, name2):

        if pandas.notnull(name1) and pandas.notnull(name2) and \
                (name1 == name2):
            return ["{}{}".format(name1, self.suffixes[0]),
                    "{}{}".format(name1, self.suffixes[1])]
        else:
            return [name1, name2]

    def fit(self):

        raise AttributeError("indexing object has no attribute 'fit'")

    def index(self, x, x_link=None):
        """Make an index of record pairs.

        Use a custom function to make record pairs of one or two dataframes.
        Each function should return a pandas.MultiIndex with record pairs.

        Parameters
        ----------
        x: pandas.DataFrame
            A pandas DataFrame. When `x_link` is None, the algorithm makes
            record pairs within the DataFrame. When `x_link` is not empty,
            the algorithm makes pairs between `x` and `x_link`.
        x_link: pandas.DataFrame, optional
            A second DataFrame to link with the DataFrame x.

        Returns
        -------
        pandas.MultiIndex
            A pandas.MultiIndex with record pairs. Each record pair contains
            the index labels of two records.

        """
        if x is None:  # error
            raise ValueError("provide at least one dataframe")
        elif x_link is not None:  # linking (two arg)
            x = (x, x_link)
        elif isinstance(x, (list, tuple)):  # dedup or linking (single arg)
            x = tuple(x)
        else:  # dedup (single arg)
            x = (x,)

        if self.verify_integrity:

            for df in x:
                self._verify_integrety(df)

        # linking
        if not self._deduplication(x):

            pairs = self._link_index(*x)
            names = self._make_index_names(x[0].index.name, x[1].index.name)

        # deduplication
        else:

            pairs = self._dedup_index(*x)
            names = self._make_index_names(x[0].index.name, x[0].index.name)

        pairs.rename(names, inplace=True)

        return pairs


BaseIndexator = DeprecationHelper(BaseIndexAlgorithm)


class BaseCompareFeature(object):
    """Base abstract class for compare feature engineering.

    Parameters
    ----------
    labels_left : list, str, int
        The labels to use for comparing record pairs in the left
        dataframe.
    labels_right : list, str, int
        The labels to use for comparing record pairs in the right
        dataframe (linking) or left dataframe (deduplication).
    args : tuple
        Additional arguments to pass to the `_compare_vectorized`
        method.
    kwargs : tuple
        Keyword additional arguments to pass to the `_compare_vectorized`
        method.
    label : list, str, int
        The identifying label(s) for the returned values.
    """

    name = None
    description = None

    def __init__(self, labels_left, labels_right, args=(), kwargs={},
                 label=None):

        self.labels_left = labels_left
        self.labels_right = labels_right
        self.args = args
        self.kwargs = kwargs
        self.label = label
        self._f_compare_vectorized = None

    def _repr(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.label)

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return repr(self)

    def _compute_vectorized(self, *args):
        """Compare attributes (vectorized)

        Parameters
        ----------
        *args : pandas.Series
            pandas.Series' as arguments.

        Returns
        -------
        pandas.Series, pandas.DataFrame, numpy.ndarray
            The result of comparing record pairs (the features). Can be
            a tuple with multiple pandas.Series, pandas.DataFrame,
            numpy.ndarray objects.

        """
        if self._f_compare_vectorized:
            return self._f_compare_vectorized(
                *(args + self.args), **self.kwargs)
        else:
            raise NotImplementedError()

    def _compute(self, left_on, right_on):
        """Compare the data on the left and right.

        :meth:`BaseCompareFeature._compute` and
        :meth:`BaseCompareFeature.compute` differ on the accepted
        arguments. `_compute` accepts indexed data while `compute`
        accepts the record pairs and the DataFrame's.

        Parameters
        ----------
        left_on : (tuple of) pandas.Series
            Data to compare with `right_on`
        right_on : (tuple of) pandas.Series
            Data to compare with `left_on`

        Returns
        -------
        pandas.Series, pandas.DataFrame, numpy.ndarray
            The result of comparing record pairs (the features). Can be
            a tuple with multiple pandas.Series, pandas.DataFrame,
            numpy.ndarray objects.
        """
        result = self._compute_vectorized(*tuple(left_on + right_on))

        return result

    def compute(self, pairs, x, x_link=None):
        """Compare the records of each record pair.

        Calling this method starts the comparing of records.

        Parameters
        ----------
        pairs : pandas.MultiIndex
            A pandas MultiIndex with the record pairs to compare. The indices
            in the MultiIndex are indices of the DataFrame(s) to link.
        x : pandas.DataFrame
            The DataFrame to link. If `x_link` is given, the comparing is a
            linking problem. If `x_link` is not given, the problem is one of
            deduplication.
        x_link : pandas.DataFrame, optional
            The second DataFrame.

        Returns
        -------
        pandas.Series, pandas.DataFrame, numpy.ndarray
            The result of comparing record pairs (the features). Can be
            a tuple with multiple pandas.Series, pandas.DataFrame,
            numpy.ndarray objects.
        """
        if not is_pandas_2d_multiindex(pairs):
            raise ValueError(
                "expected pandas.MultiIndex with record pair indices "
                "as first argument"
            )

        if not isinstance(x, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as second argument")

        if x_link is not None and not isinstance(x_link, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as third argument")

        labels_left = listify(self.labels_left, [])
        labels_right = listify(self.labels_right, [])

        if x_link is None:
            df_a = frame_indexing(x[labels_left + labels_right], pairs, 0)
            data1 = tuple([df_a[lbl] for lbl in listify(self.labels_left)])
            data2 = tuple([df_a[lbl] for lbl in listify(self.labels_right)])
        else:
            df_a = frame_indexing(x[labels_left], pairs, 0)
            data1 = tuple([df_a[lbl] for lbl in listify(self.labels_left)])
            df_b = frame_indexing(x_link[labels_right], pairs, 1)
            data2 = tuple([df_b[lbl] for lbl in listify(self.labels_right)])

        results = self._compute(data1, data2)

        return results


class BaseCompare(object):
    """Base class for all comparing classes in Python Record Linkage Toolkit.

    Parameters
    ----------
    features : list
        List of compare algorithms.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for comparing of record pairs.
        If -1, then the number of jobs is set to the number of cores.
    indexing_type : string, optional (default='label')
        The indexing type. The MultiIndex is used to index the DataFrame(s).
        This can be done with pandas ``.loc`` or with ``.iloc``. Use the value
        'label' to make use of ``.loc`` and 'position' to make use of
        ``.iloc``. The value 'position' is only available when the MultiIndex
        consists of integers. The value 'position' is much faster.

    Attributes
    ----------
    features: list
        A list of algorithms to create features.

    """

    def __init__(self, features=[], n_jobs=1, indexing_type='label',
                 **kwargs):

        logging.info("comparing - initialize {} class".format(
            self.__class__.__name__)
        )

        self.features = []
        self.add(features)

        # public
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        self.indexing_type = indexing_type  # label of position

        # logging
        self._i = 1
        self._i_max = None
        self._n = []
        self._eta = []
        self._output_log_total = True

        # private
        self._compare_functions = []

        if isinstance(features, (pandas.MultiIndex, pandas.Index)):
            warnings.warn(
                "It seems you are using the older version of the Compare API, "
                "see the documentation about how to update to the new API. "
                "http://recordlinkage.readthedocs.io/"
                "en/latest/ref-compare.html",
                DeprecationWarning
            )

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{}>".format(class_name)

    def __str__(self):
        return repr(self)

    def add(self, model):
        """Add a compare method.

        This method is used to add compare features.

        Parameters
        ----------
        model : list, class
            A (list of) compare feature(s) from
            :mod:`recordlinkage.compare`.
        """
        if isinstance(model, list):
            self.features.extend(model)
        else:
            self.features.append(model)

    def compare_vectorized(self, comp_func, labels_left, labels_right,
                           *args, **kwargs):
        """Compute the similarity between values with a callable.

        This method initialises the comparing of values with a custom
        function/callable. The function/callable should accept
        numpy.ndarray's.

        Example
        -------

        >>> comp = recordlinkage.Compare()
        >>> comp.compare_vectorized(custom_callable, 'first_name', 'name')
        >>> comp.compare(PAIRS, DATAFRAME1, DATAFRAME2)

        Parameters
        ----------
        comp_func : function
            A comparison function. This function can be a built-in function
            or a user defined comparison function. The function should accept
            numpy.ndarray's as first two arguments.
        labels_left : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        labels_right : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        *args :
            Additional arguments to pass to callable comp_func.
        **kwargs :
            Additional keyword arguments to pass to callable comp_func.
            (keyword 'label' is reserved.)
        label : (list of) label(s)
            The name of the feature and the name of the column. IMPORTANT:
            This argument is a keyword argument and can not be part of the
            arguments of comp_func.
        """
        label = kwargs.pop('label', None)

        if isinstance(labels_left, tuple):
            labels_left = list(labels_left)

        if isinstance(labels_right, tuple):
            labels_right = list(labels_right)

        feature = BaseCompareFeature(
            labels_left, labels_right, args, kwargs, label=label)
        feature._f_compare_vectorized = comp_func

        self.add(feature)

    def _get_labels_left(self, validate=None):
        """Get all labels of the left dataframe."""

        labels = []

        for compare_func in self.features:

            labels = labels + listify(compare_func.labels_left)

        # check requested labels (for better error messages)
        if not is_label_dataframe(labels, validate):
            error_msg = "label is not found in the dataframe"
            raise KeyError(error_msg)

        return unique(labels)

    def _get_labels_right(self, validate=None):
        """Get all labels of the right dataframe."""
        labels = []

        for compare_func in self.features:

            labels = labels + listify(compare_func.labels_right)

        # check requested labels (for better error messages)
        if not is_label_dataframe(labels, validate):
            error_msg = "label is not found in the dataframe"
            raise KeyError(error_msg)

        return unique(labels)

    def _compute_parallel(self, pairs, x, x_link=None, n_jobs=1):

        df_chunks = index_split(pairs, n_jobs)
        result_chunks = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_compare_helper)(self, chunk, x, x_link)
            for chunk in df_chunks
        )

        result = pandas.concat(result_chunks)
        return result

    def _compute(self, pairs, x, x_link=None):

        # start the timer for the comparing step
        start_time = time.time()

        sublabels_left = self._get_labels_left(validate=x)
        df_a_indexed = frame_indexing(x[sublabels_left], pairs, 0)

        if x_link is None:
            sublabels_right = self._get_labels_right(validate=x)
            df_b_indexed = frame_indexing(x[sublabels_right], pairs, 1)
        else:
            sublabels_right = self._get_labels_right(validate=x_link)
            df_b_indexed = frame_indexing(x_link[sublabels_right], pairs, 1)

        # log timing
        # index_time = time.time() - start_time

        features = []

        for feat in self.features:

            # --- DATA1
            # None: no data passed to func
            if feat.labels_left is None:
                data1 = tuple()
            # empty array: empty df with index passed to func
            elif feat.labels_left == []:
                data1 = (df_a_indexed[[]],)
            # else: subset columns and pass tuple of series
            else:
                data1 = tuple(
                    [df_a_indexed[lbl] for lbl in listify(feat.labels_left)]
                )

            # --- DATA2
            # None: no data passed to func
            if feat.labels_right is None:
                data2 = tuple()
            # empty array: empty df with index passed to func
            elif feat.labels_right == []:
                data2 = (df_b_indexed[[]],)
            # else: subset columns and pass tuple of series
            else:
                data2 = tuple(
                    [df_b_indexed[lbl] for lbl in listify(feat.labels_right)]
                )

            result = feat._compute(data1, data2)
            features.append((result, feat.label))

        features = self._union(features, pairs)

        # log timing
        n = pairs.shape[0]
        i_max = '?' if self._i_max is None else self._i_max
        eta = time.time() - start_time
        self._eta.append(eta)
        self._n.append(n)

        # log
        logging.info("comparing [{:d}/{}] - time: {:.2f}s - pairs: {}".format(
            self._i, i_max, eta, n))

        # log total
        if self._output_log_total:

            n_total = np.sum(self._n)
            eta_total = np.sum(self._eta)

            logging.info(
                "comparing [{:d}/{}] - time: {:.2f}s - pairs_total: {}".format(
                    self._i, i_max, eta_total, n_total))

        self._i += 1

        return features

    def _union(self, objs, index=None, column_i=0):
        """Make a union of the features.

        The term 'union' is based on the terminology of scikit-learn.
        """
        feat_conc = []

        for feat, label in objs:

            # result is tuple of results
            if isinstance(feat, tuple):
                if label is None:
                    label = [None] * len(feat)
                if not is_list_like(label):
                    label = (label,)

                partial_result = self._union(
                    zip(feat, label), column_i=column_i)
                feat_conc.append(partial_result)
                column_i = column_i + partial_result.shape[1]

            # result is pandas.Series.
            elif isinstance(feat, pandas.Series):
                feat.reset_index(drop=True, inplace=True)
                if label is None:
                    label = column_i
                feat.rename(label, inplace=True)
                feat_conc.append(feat)
                column_i = column_i + 1

            # result is pandas.DataFrame
            elif isinstance(feat, pandas.DataFrame):
                feat.reset_index(drop=True, inplace=True)
                if label is None:
                    label = np.arange(column_i, column_i + feat.shape[1])
                feat.columns = label
                feat_conc.append(feat)
                column_i = column_i + feat.shape[1]

            # result is numpy 1d array
            elif is_numpy_like(feat) and len(feat.shape) == 1:
                if label is None:
                    label = column_i
                f = pandas.Series(feat, name=label, copy=False)

                feat_conc.append(f)
                column_i = column_i + 1

            # result is numpy 2d array
            elif is_numpy_like(feat) and len(feat.shape) == 2:
                if label is None:
                    label = np.arange(column_i, column_i + feat.shape[1])
                feat_df = pandas.DataFrame(feat, columns=label, copy=False)
                if label is None:
                    feat_df.columns = [None for _ in range(feat_df.shape[1])]
                feat_conc.append(feat_df)
                column_i = column_i + feat.shape[1]

            # other results are not (yet) supported
            else:
                raise ValueError("expected numpy.ndarray or "
                                 "pandas object to be returned, "
                                 "got '{}'".format(feat.__class__.__name__))

        result = pandas.concat(feat_conc, axis=1, copy=False)
        if index is not None:
            result.set_index(index, inplace=True)

        return result

    def compute(self, pairs, x, x_link=None):
        """Compare the records of each record pair.

        Calling this method starts the comparing of records.

        Parameters
        ----------
        pairs : pandas.MultiIndex
            A pandas MultiIndex with the record pairs to compare. The indices
            in the MultiIndex are indices of the DataFrame(s) to link.
        x : pandas.DataFrame
            The DataFrame to link. If `x_link` is given, the comparing is a
            linking problem. If `x_link` is not given, the problem is one of
            deduplication.
        x_link : pandas.DataFrame, optional
            The second DataFrame.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame with feature vectors, i.e. the result of
            comparing each record pair.
        """
        if not isinstance(pairs, pandas.MultiIndex):
            raise ValueError(
                "expected pandas.MultiIndex with record pair indices "
                "as first argument"
            )

        if not isinstance(x, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as second argument")

        if x_link is not None and not isinstance(x_link, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as third argument")

        if self.n_jobs == 1:
            results = self._compute(pairs, x, x_link)
        elif self.n_jobs > 1:
            results = self._compute_parallel(
                pairs, x, x_link, n_jobs=self.n_jobs)
        else:
            raise ValueError("number of jobs should be positive integer")

        return results

    def compare(self, *args, **kwargs):
        """[DEPRECATED] Compare two records."""

        raise AttributeError("this method was removed in version 0.12.0")

    def clear_memory(self):
        """[DEPRECATED] Clear memory."""

        raise AttributeError("this method was removed in version 0.12.0")


class BaseClassifier(metaclass=ABCMeta):
    """Base class for classification of records pairs.

    This class contains methods for training the classifier.
    Distinguish different types of training, such as supervised and
    unsupervised learning.
    """
    def __init__(self):
        pass

    def learn(self, *args, **kwargs):
        """[DEPRECATED] Use 'fit_predict'.
        """

        warnings.warn("learn is deprecated, {}.fit_predict "
                      "instead".format(self.__class__.__name__))
        return self.fit_predict(*args, **kwargs)

    def _initialise_classifier(self, comparison_vectors):
        """Initialise the classifier.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The comparison vectors (or features) to fit the classifier with.
        """
        pass

    @abstractmethod
    def _fit(self, *args, **kwargs):
        pass

    def fit(self, comparison_vectors, match_index=None):
        """Train the classifier.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The comparison vectors (or features) to train the model with.
        match_index : pandas.MultiIndex
            A pandas.MultiIndex object with the true matches.
            The MultiIndex contains only the true matches. Default None.

        Note
        ----

        A note in case of finding links within a single dataset (for example
        deduplication). Unsure that the training record pairs are from the
        lower triangular part of the dataset/matrix. See detailed information
        here: link.

        """
        logging.info("Classification - start training {}".format(
            self.__class__.__name__)
        )

        self._initialise_classifier(comparison_vectors)

        # start timing
        start_time = time.time()

        if isinstance(match_index, (pandas.MultiIndex, pandas.Index)):

            try:
                y = pandas.Series(0, index=comparison_vectors.index)
                y.loc[match_index & comparison_vectors.index] = 1
            except pandas.IndexError as err:

                # The are no matches. So training is not possible.
                if len(match_index & comparison_vectors.index) == 0:
                    raise LearningError(
                        "both matches and non-matches needed in the" +
                        "trainingsdata, only non-matches found"
                    )
                else:
                    raise err

            self._fit(comparison_vectors.values, y.values)

        elif match_index is None:
            self._fit(comparison_vectors.values)
        else:
            raise ValueError(
                "'match_index' has incorrect type '{}'".format(
                    type(match_index)
                )
            )

        # log timing
        logf_time = "Classification - training computation time: ~{:.2f}s"
        logging.info(logf_time.format(time.time() - start_time))

    @return_type_deprecator
    def fit_predict(self, comparison_vectors, match_index=None):
        """Train the classifier.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The comparison vectors.
        match_index : pandas.MultiIndex
            The true matches.
        return_type : str
            Deprecated. Use recordlinkage.options instead. Use the option
            `recordlinkage.set_option('classification.return_type', 'index')`
            instead.

        Returns
        -------
        pandas.Series
            A pandas Series with the labels 1 (for the matches) and 0 (for the
            non-matches).

        """
        self.fit(comparison_vectors, match_index)
        result = self.predict(comparison_vectors)

        return result

    @abstractmethod
    def _predict(self, comparison_vectors):
        pass

    @return_type_deprecator
    def predict(self, comparison_vectors):
        """Predict the class of the record pairs.

        Classify a set of record pairs based on their comparison vectors into
        matches, non-matches and possible matches. The classifier has to be
        trained to call this method.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            Dataframe with comparison vectors.
        return_type : str
            Deprecated. Use recordlinkage.options instead. Use the option
            `recordlinkage.set_option('classification.return_type', 'index')`
            instead.

        Returns
        -------
        pandas.Series
            A pandas Series with the labels 1 (for the matches) and 0 (for the
            non-matches).

        """
        logging.info("Classification - predict matches and non-matches")

        # make the predicition
        prediction = self._predict(comparison_vectors.values)
        self._post_predict(prediction)

        # format and return the result
        return self._return_result(prediction, comparison_vectors)

    def _post_predict(self, result):
        """Method called after prediction.

        Parameters
        ----------
        result : pandas.Series
            The resulting classification.
        """
        pass

    @abstractmethod
    def _prob_match(self, *args, **kwargs):
        pass

    def prob(self, comparison_vectors, return_type=None):
        """Compute the probabilities for each record pair.

        For each pair of records, estimate the probability of being a match.

        Parameters
        ----------
        comparison_vectors : pandas.DataFrame
            The dataframe with comparison vectors.
        return_type : str
            Deprecated. (default 'series')

        Returns
        -------
        pandas.Series or numpy.ndarray
            The probability of being a match for each record pair.

        """
        if return_type is not None:
            warnings.warn("The argument 'return_type' is removed. "
                          "Default value is now 'series'.",
                          DeprecationWarning, stacklevel=2)

        logging.info("Classification - compute probabilities")

        prob_match = self._prob_match(comparison_vectors.values)
        return pandas.Series(prob_match, index=comparison_vectors.index)

    def _return_result(self, result, comparison_vectors=None):
        """Return different formatted classification results.

        """
        return_type = cf.get_option('classification.return_type')

        if type(result) != np.ndarray:
            raise ValueError("numpy.ndarray expected.")

        # return the pandas.MultiIndex
        if return_type == 'index':
            return comparison_vectors.index[result.astype(bool)]

        # return a pandas.Series
        elif return_type == 'series':
            return pandas.Series(
                result,
                index=comparison_vectors.index,
                name='classification')

        # return a numpy.ndarray
        elif return_type == 'array':
            return result

        # return_type not known
        else:
            raise ValueError(
                "return_type {} unknown. Choose 'index', 'series' or "
                "'array'".format(return_type))
