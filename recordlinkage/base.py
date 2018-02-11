"""Base module for record linkage."""

from __future__ import division

import time
import warnings

import pandas
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from recordlinkage.utils import (listify,
                                 unique,
                                 is_label_dataframe,
                                 VisibleDeprecationWarning,
                                 index_split,
                                 frame_indexing)
from recordlinkage.types import (is_string_like,
                                 is_pandas_like,
                                 is_pandas_2d_multiindex)
from recordlinkage.measures import max_pairs
from recordlinkage import rl_logging as logging


def chunk_pandas(frame_or_series, chunksize=None):
    """Chunk a frame into smaller, equal parts."""

    if not isinstance(chunksize, int):
        raise ValueError('argument chunksize needs to be integer type')

    bins = np.arange(0, len(frame_or_series), step=chunksize)

    for b in bins:

        yield frame_or_series[b:b + chunksize]


class BaseIndexator(object):
    """Base class for all indexator classes in Python Record Linkage Toolkit.

    The structure of the indexing classes follow the framework of SciKit-
    learn and tensorflow.

    Example
    -------
    Make your own indexation class
    ```
    class CustomIndex(BaseIndexator):

        def _link_index(self, df_a, df_b):

            # Custom link index.

            return ...

        def _dedup_index(self, df_a):

            # Custom deduplication index, optional.

            return ...
    ```

    Call the class in the same way
    ```
    custom_index = CustomIndex():
    custom_index.index()
    ```
    """

    def __init__(self, verify_integrity=True, suffixes=('_1', '_2')):
        super(BaseIndexator, self).__init__()

        self.suffixes = suffixes
        self.verify_integrity = verify_integrity

        self._n = []
        self._n_max = []

        logging.info("Indexing - initialize {} class".format(
            self.__class__.__name__)
        )

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{}>".format(class_name)

    def __str__(self):
        return repr(self)

    @classmethod
    def _deduplication(cls, x):

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

        return NotImplementedError(
            "Not possible to call index for the BaseEstimator"
        )

    def _dedup_index(self, df_a):
        """Make record pairs of a single dataframe."""

        pairs = self._link_index(df_a, df_a)

        # Remove all pairs not in the upper triangular part of the matrix.
        # This part can be inproved by not comparing the level values, but the
        # level itself.
        pairs = pairs[
            pairs.get_level_values(0) < pairs.get_level_values(1)
        ]

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

        # start timing
        start_time = time.time()

        # linking
        if not self._deduplication(x):
            logging.info("Indexing - start indexing two DataFrames")

            pairs = self._link_index(*x)
            names = self._make_index_names(x[0].index.name, x[1].index.name)

        # deduplication
        else:
            logging.info("Indexing - start indexing single DataFrame")

            pairs = self._dedup_index(*x)
            names = self._make_index_names(x[0].index.name, x[0].index.name)

        pairs.rename(names, inplace=True)

        # store the number of pairs
        self._n.append(pairs.shape[0])
        self._n_max.append(max_pairs(x))

        # summary
        n = len(pairs)
        rr = 1 - self._n[-1] / self._n_max[-1]
        rr_avg = 1 - np.sum(self._n) / np.sum(self._n_max)

        # log timing
        logf_time = "Indexing - computation time: ~{:.2f}s"
        logging.info(logf_time.format(time.time() - start_time))

        # log results
        logf_result = "Indexing - summary n={:d}, " \
            "reduction_ratio={:0.5f}, reduction_ratio_mean={:0.5f}"
        logging.info(logf_result.format(n, rr, rr_avg))

        return pairs


class BaseCompareFeature(object):
    """BaseCompareFeature construction class.
    """

    name = None
    description = None

    def __init__(self, labels_left, labels_right, args=(),
                 kwargs={}):

        self.labels_left = labels_left
        self.labels_right = labels_right
        self.args = args
        self.kwargs = kwargs
        self._f_compare_vectorized = None

        # logging
        logging.info(
            "{} - initialize exact algorithm "
            "- compare {l_left} with {l_right}".format(
                self.__class__.__name__,
                l_left=labels_left,
                l_right=labels_right)
        )

    def _repr(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.label)

    def __repr__(self):
        return self._repr()

    def __str__(self):
        return repr(self)

    def _compute_vectorized(self, *args):
        """Compare attributes (vectorized)"""

        if self._f_compare_vectorized:
            return self._f_compare_vectorized(
                *(args + self.args), **self.kwargs)
        else:
            raise NotImplementedError()

    def _compute_single(self):
        """Compare attributes (non-vectorized)"""

        raise NotImplementedError()

    def _compute(self, *args):

        logging.info("Comparing - start comparing data")

        # start the timer for the comparing step
        start_time = time.time()

        c = self._compute_vectorized(*args)

        # log timing
        total_time = time.time() - start_time

        # log timing
        logging.info(
            "Comparing - computation time: ~{:.2f}s".format(total_time))

        # log results
        logf_result = "Comparing - summary shape={}"
        logging.info(logf_result.format(c.shape))

        return c

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

        if not is_pandas_2d_multiindex(pairs):
            raise ValueError(
                "expected pandas.MultiIndex with record pair indices "
                "as first argument"
            )

        if not isinstance(x, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as second argument")

        if x_link is not None and not isinstance(x_link, pandas.DataFrame):
            raise ValueError("expected pandas.DataFrame as third argument")

        df_a_indexed = frame_indexing(x[self.labels_left], pairs, 0)

        if x_link is None:
            df_b_indexed = frame_indexing(x[self.labels_right], pairs, 1)
        else:
            df_b_indexed = frame_indexing(x_link[self.labels_right], pairs, 1)

        results = self._compute(df_a_indexed, df_b_indexed)

        return results


class BaseCompare(object):
    """Base class for all comparing classes in Python Record Linkage Toolkit.
    """

    def __init__(self, pairs=None, df_a=None, df_b=None, low_memory=False,
                 block_size=1000000, n_jobs=1, indexing_type='label',
                 **kwargs):

        logging.info("Comparing - initialize {} class".format(
            self.__class__.__name__)
        )

        # public
        self.n_jobs = n_jobs
        self.indexing_type = indexing_type  # label of position
        self.features = []

        # private
        self._compare_functions = []

        if isinstance(pairs, (pandas.MultiIndex, pandas.Index)):
            warnings.warn(
                "It seems you are using the older version of the Compare API, "
                "see the documentation about how to update to the new API. "
                "http://recordlinkage.readthedocs.io/"
                "en/latest/ref-compare.html",
                VisibleDeprecationWarning
            )

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{}>".format(class_name)

    def __str__(self):
        return repr(self)

    def add(self, model, label=None):

        self.features.append((model, label))

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
            labels_left, labels_right, args, kwargs)
        feature._f_compare_vectorized = comp_func

        self.add(feature, label=label)

    def _get_labels_left(self, validate=None):
        """Get all labels of the left dataframe."""

        labels = []

        for compare_func, _ in self.features:

            labels = labels + listify(compare_func.labels_left)

        # check requested labels (for better error messages)
        if not is_label_dataframe(labels, validate):
            error_msg = "label is not found in the dataframe"
            raise KeyError(error_msg)

        return unique(labels)

    def _get_labels_right(self, validate=None):
        """Get all labels of the right dataframe."""
        labels = []

        for compare_func, _ in self.features:

            labels = labels + listify(compare_func.labels_right)

        # check requested labels (for better error messages)
        if not is_label_dataframe(labels, validate):
            error_msg = "label is not found in the dataframe"
            raise KeyError(error_msg)

        return unique(labels)

    def _compute_parallel(self, pairs, x, x_link=None, n_jobs=1):

        df_chunks = index_split(pairs, n_jobs)
        result_chunks = Parallel(n_jobs=n_jobs)(
            delayed(self._compute)(chunk, x, x_link) for chunk in df_chunks
        )

        result = pandas.concat(result_chunks)
        return result

    def _compute(self, pairs, x, x_link=None):

        logging.info("Comparing - start comparing data")

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
        index_time = time.time() - start_time

        results = pandas.DataFrame(index=pairs)
        label_num = 0  # make a label is label is None

        for feat, label in self.features:

            lbl1 = feat.labels_left
            lbl2 = feat.labels_right

            data1 = tuple([df_a_indexed[lbl] for lbl in listify(lbl1)])
            data2 = tuple([df_b_indexed[lbl] for lbl in listify(lbl2)])

            c = feat._compute(*tuple(data1 + data2))

            if is_pandas_like(c):
                c = c.values  # convert pandas into numpy

            if label is not None:
                label = listify(label)

            n_cols = 1 if len(c.shape) == 1 else c.shape[1]

            labels = []
            for i in range(0, n_cols):

                label_val = label[i] if label is not None else label_num
                label_num += 1

                labels.append(label_val)

            results[label_val] = c

        # log timing
        total_time = time.time() - start_time

        # log timing
        logging.info("Comparing - computation time: ~{:.2f}s (from which "
                     "indexing: ~{:.2f}s)".format(total_time, index_time))

        # log results
        logf_result = "Comparing - summary shape={}"
        logging.info(logf_result.format(results.shape))

        return results

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
