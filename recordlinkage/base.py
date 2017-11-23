"""Base module for record linkage."""

from __future__ import division

import time
import warnings

import pandas
import numpy as np


from recordlinkage.utils import listify
from recordlinkage.utils import unique
from recordlinkage.utils import is_label_dataframe
from recordlinkage.utils import VisibleDeprecationWarning

from recordlinkage.types import is_pandas_like
from recordlinkage.types import is_numpy_like

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

    def __init__(self, verify_integrity=True):
        super(BaseIndexator, self).__init__()

        self._n = []
        self._n_max = []

        self.verify_integrity = verify_integrity

        logging.info("Indexing - initialize {} class".format(
            self.__class__.__name__)
        )

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

        # deduplication
        else:
            logging.info("Indexing - start indexing single DataFrame")

            pairs = self._dedup_index(*x)

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


class BaseCompare(object):
    """Base class for all comparing classes in Python Record Linkage Toolkit.

    Class to compare the attributes of candidate record pairs. The ``Compare2``
    class has several methods to compare data such as string similarity
    measures, numeric metrics and exact comparison methods. The ``Compare2``
    class has an improved API desing compares to ``Compare`` class. The
    ``Compare`` class will get deprecated in the next version. After this,
    the ``Compare2`` class will be renamed into ``Compare``.

    Parameters
    ----------
    pairs : pandas.MultiIndex (DEPRECATED)
        A MultiIndex of candidate record pairs.
    df_a : pandas.DataFrame (DEPRECATED)
        The first dataframe.
    df_b : pandas.DataFrame (DEPRECATED)
        The second dataframe.
    low_memory : bool (DEPRECATED)
        Reduce the amount of memory used by the Compare class. Default False.
    block_size : int
        The maximum size of data blocks. Default 1,000,000.

    Examples
    --------
    In the following example, the record pairs of two historical datasets with
    census data are compared. The datasets are named ``census_data_1980`` and
    ``census_data_1990``. The ``candidate_pairs`` are the record pairs to
    compare. The record pairs are compared on the first name, last name, sex,
    date of birth, address, place, and income.

    >>> comp = recordlinkage.Compare()
    >>> comp.string('first_name', 'name', method='jarowinkler')
    >>> comp.string('lastname', 'lastname', method='jarowinkler')
    >>> comp.exact('dateofbirth', 'dob')
    >>> comp.exact('sex', 'sex')
    >>> comp.string('address', 'address', method='levenshtein')
    >>> comp.exact('place', 'place')
    >>> comp.numeric('income', 'income')
    >>> comp.compute(candidate_pairs, census_data_1980, census_data_1990)

    The attribute ``vectors`` is the DataFrame with the comparison data. It
    can be called whenever you want.
    """

    def __init__(self, pairs=None, df_a=None, df_b=None, low_memory=False,
                 block_size=1000000, njobs=1, indexing_type='label', **kwargs):

        logging.info("Comparing - initialize {} class".format(
            self.__class__.__name__)
        )

        # public
        self.indexing_type = indexing_type  # label of position

        # private
        self._compare_functions = []

        if isinstance(pairs, (pandas.MultiIndex, pandas.Index)):
            self.deprecated = True

            warnings.warn(
                "It seems you are using the older version of the Compare API, "
                "see the documentation about how to update to the new API. "
                "http://recordlinkage.readthedocs.io/"
                "en/latest/ref-compare.html",
                VisibleDeprecationWarning
            )
        else:
            self.deprecated = False

        # deprecated
        self.df_a = df_a
        self.df_b = df_b if df_b is not None else df_a

        self.pairs = pairs

        self.low_memory = low_memory
        self.block_size = block_size
        self.njobs = njobs

        self._df_a_indexed = None
        self._df_b_indexed = None

        self.vectors = pandas.DataFrame(index=pairs)

    def _loc2(self, frame, multi_index, level_i, indexing_type='label'):
        """Indexing algorithm for MultiIndex on one level

        Arguments
        ---------
        frame : pandas.DataFrame
            The datafrme to select records from.
        multi_index : pandas.MultiIndex
            A pandas multiindex were one fo the levels is used to sample the
            dataframe with.
        level_i : int, str
            The level of the multiIndex to index on.
        indexing_type : str
            The type of indexing. The value can be 'label' or 'position'.
            Default 'label'.

        """

        if indexing_type == "label":
            data = frame.loc[multi_index.get_level_values(level_i)]
            data.index = multi_index
        elif indexing_type == "position":
            data = frame.iloc[multi_index.get_level_values(level_i)]
            data.index = multi_index
        else:
            raise ValueError("indexing_type needs to be 'label' or 'position'")

        return data

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
            This argument is a keyword argument.
        """

        log_str = "Comparing - initialize user defined function - " \
            "compare {l_left} with {l_right}"
        logging.info(log_str.format(l_left=labels_left, l_right=labels_right))

        return self._compare_vectorized(
            comp_func, labels_left, labels_right, *args, **kwargs)

    def _compare_vectorized(self, comp_func, labels_left, labels_right,
                            *args, **kwargs):

        # Use recordlinkage >=0.10.0
        if not self.deprecated:

            label = kwargs.pop('label', None)

            if isinstance(labels_left, tuple):
                labels_left = list(labels_left)

            if isinstance(labels_right, tuple):
                labels_right = list(labels_right)

            # all args and kwargs are passed to comp_func
            self._compare_functions.append(
                (comp_func, labels_left, labels_right, label, args, kwargs,)
            )

            return self

        # Use recordlinkage < 0.10.0
        else:

            return self.compare(
                comp_func, labels_left, labels_right, *args, **kwargs)

    def _get_labels(self, frame_i, validate=None):
        """Get all labels.

        Parameters
        ----------
        frame_i : str
            A string, 'left' or 'right', incidating the dataframe to collect
            labels from.

        """

        labels = []

        for compare_func in self._compare_functions:

            labels = labels + listify(compare_func[frame_i])

        # check requested labels (for better error messages)
        if not is_label_dataframe(labels, validate):
            error_msg = "label is not found in the dataframe"
            raise KeyError(error_msg)

        return unique(labels)

    def _get_labels_left(self, validate=None):
        """Get all labels of the left dataframe."""

        return self._get_labels(1, validate=validate)

    def _get_labels_right(self, validate=None):
        """Get all labels of the right dataframe."""

        return self._get_labels(2, validate=validate)

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

        logging.info("Comparing - start comparing data")

        # start the timer for the comparing step
        start_time = time.time()

        sublabels_left = self._get_labels_left(validate=x)
        df_a_indexed = self._loc2(x[sublabels_left], pairs, 0)

        if x_link is None:
            sublabels_right = self._get_labels_right(validate=x)
            df_b_indexed = self._loc2(x[sublabels_right], pairs, 1)
        else:
            sublabels_right = self._get_labels_right(validate=x_link)
            df_b_indexed = self._loc2(x_link[sublabels_right], pairs, 1)

        # log timing
        index_time = time.time() - start_time

        results = pandas.DataFrame(index=pairs)
        label_num = 0  # make a label is label is None

        for f, lbl1, lbl2, label, args, kwargs in self._compare_functions:

            data1 = tuple([df_a_indexed[lbl] for lbl in listify(lbl1)])
            data2 = tuple([df_b_indexed[lbl] for lbl in listify(lbl2)])

            c = f(*tuple(data1 + data2 + args), **kwargs)

            if isinstance(c, (pandas.Series, pandas.DataFrame)):
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

    def compare(self, comp_func, labels_a, labels_b, *args, **kwargs):
        """[DEPRECATED] Compare two records.

        Core method to compare record pairs. This method takes a function and
        data from both records in the record pair. The data is compared with
        the compare function. The built-in methods also use this function.

        Example
        -------

        >>> comp = recordlinkage.Compare(PAIRS, DATAFRAME1, DATAFRAME2)
        >>> comp.exact('first_name', 'name')

        >>> # same as
        >>> comp.compare(recordlinkage._compare_exact, 'first_name', 'name')

        Parameters
        ----------
        comp_func : function
            A comparison function. This function can be a built-in function or
            a user defined comparison function.
        labels_a : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        labels_b : label, pandas.Series, pandas.DataFrame
            The labels, Series or DataFrame to compare.
        name : label
            The name of the feature and the name of the column.
        store : bool, default True
            Store the result in the dataframe.

        Returns
        -------
        pandas.Series
            A pandas series with the result of comparing each record pair.

        """

        if isinstance(comp_func, pandas.MultiIndex):
            raise ValueError("see new api documentation: "
                             "use method 'compute' instead of 'compare'")

        if len(self.pairs) == 0:
            raise ValueError(
                "need at least one record pair"
            )

        # the name and store arguments
        name = kwargs.pop('name', None)
        store = kwargs.pop('store', True)

        labels_a = listify(labels_a)
        labels_b = listify(labels_b)

        data_a = []

        for label_a in labels_a:

            # the label is a numpy or pandas object
            if is_numpy_like(label_a) or is_pandas_like(label_a):
                data_a.append(label_a)

            # check requested labels (for better error messages)
            elif label_a not in self.df_a.columns:
                raise KeyError("label '{}' is not found in the first"
                               "dataframe".format(label_a))

            else:

                if self.low_memory:

                    df_a_label = self._loc2(self.df_a[label_a], self.pairs, 0)
                    data_a.append(df_a_label)

                # not low memory
                else:
                    if self._df_a_indexed is None:

                        self._df_a_indexed = self._loc2(
                            self.df_a, self.pairs, 0)

                    data_a.append(self._df_a_indexed[label_a])

        data_a = tuple(data_a)

        data_b = []

        for label_b in labels_b:

            # the label is a numpy or pandas object
            if is_numpy_like(label_b) or is_pandas_like(label_b):
                data_b.append(label_b)

            # check requested labels (for better error messages)
            elif label_b not in self.df_b.columns:

                raise KeyError("label '{}' is not found in the second"
                               "dataframe".format(label_b))

            else:

                if self.low_memory:

                    df_b_label = self._loc2(self.df_b[label_b], self.pairs, 1)
                    data_b.append(df_b_label)

                # not low memory
                else:
                    if self._df_b_indexed is None:

                        self._df_b_indexed = self._loc2(
                            self.df_b, self.pairs, 1)

                    data_b.append(self._df_b_indexed[label_b])

        data_b = tuple(data_b)

        # Compute the comparison
        c = comp_func(*tuple(data_a + data_b + args), **kwargs)

        # if a pandas series is returned, overwrite the index. The
        # returned index can be different than the MultiIndex passed to
        # the compare function.
        if isinstance(c, pandas.Series):
            c.index = self.vectors.index

        # append column to Compare.vectors
        if store:
            name_or_id = name if name else len(self.vectors.columns)
            self.vectors[name_or_id] = c

        return self.vectors[name_or_id].rename(name)

    def clear_memory(self):
        """[DEPRECATED] Clear memory.

        Clear some memory when low_memory was set to True.
        """

        self._df_a_indexed = None
        self._df_b_indexed = None
