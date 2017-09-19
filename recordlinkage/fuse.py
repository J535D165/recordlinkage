from __future__ import division
from __future__ import unicode_literals

import types
import typing
import datetime
import multiprocessing as mp
from typing import Callable

import pandas as pd

from recordlinkage import rl_logging
from recordlinkage.utils import listify
from recordlinkage.algorithms.conflict_resolution import (annotated_concat,
                                                          choose_trusted,
                                                          choose_first,
                                                          choose_last,
                                                          choose_max,
                                                          choose_min,
                                                          choose_shortest,
                                                          choose_longest,
                                                          choose_shortest_tie_break,
                                                          choose_longest_tie_break,
                                                          choose_metadata_max,
                                                          choose_metadata_min,
                                                          choose_random,
                                                          aggregate,
                                                          count,
                                                          group,
                                                          identity,
                                                          no_gossip,
                                                          vote,
                                                          nullify)


def process_tie_break(tie_break) -> Callable[[tuple, bool], any]:
    """
    Handles string/function -> function happing for tie breaking.

    :param str/function tie_break: A conflict resolution function to be used in the case of a tie. May be a
    conflict resolution function or a string (one of random, trust_a, trust_b, min, max, shortest, longest, or null.
    :return: A function
    """
    tie_break_fun = None
    if tie_break is None:
        pass
    elif isinstance(tie_break, types.FunctionType):
        tie_break_fun = tie_break
    elif isinstance(tie_break, str):
        if tie_break == 'random':
            tie_break_fun = choose_random
        elif tie_break == 'trust_a':
            tie_break_fun = choose_first
        elif tie_break == 'trust_b':
            tie_break_fun = choose_last
        elif tie_break == 'min':
            tie_break_fun = choose_min
        elif tie_break == 'max':
            tie_break_fun = choose_max
        elif tie_break == 'shortest':
            tie_break_fun = choose_shortest_tie_break
        elif tie_break == 'longest':
            tie_break_fun = choose_longest_tie_break
        elif tie_break == 'null':
            tie_break_fun = nullify
        else:
            raise ValueError('Invalid tie_break strategy: {}. Must be one of'
                             'random, trust_a, trust_b, min, max, shortest, longest, or null.'.format(tie_break))
    else:
        raise ValueError('tie_break must be a string or a function.')
    return tie_break_fun


class FuseCore(object):
    def __init__(self):
        """
        ``FuseCore`` and its subclasses are initialized without data. The initialized
        object is populated by metadata describing a series of data resolutions,
        which are executed when ``.fuse()`` is called.
        """
        self.vectors = None
        self.index = None
        self.predictions = None
        self.df_a = None
        self.df_b = None
        self.suffix_a = None
        self.suffix_b = None
        self.resolution_queue = []
        self._bases_taken = []
        self._names_taken = []
        self._sep = ''
        self._index_level_0 = None
        self._index_level_1 = None

    # Conflict Resolution Realizations

    def no_gossiping(self, values_a, values_b, name=None):
        """
        Handles data conflicts by keeping values agree upon by both data sources,
        and returning np.nan for conflicting or missing values.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str name: The name of the resolved column.
        :return: None
        """
        self.resolve(no_gossip, values_a, values_b, name=name, remove_na_vals=False, description='no_gossiping')

    def roll_the_dice(self, values_a, values_b, name=None, remove_na_vals=True):
        """
        Handles data conflicts by choosing a random non-missing value.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :return: None
        """
        self.resolve(choose_random, values_a, values_b, name=name, remove_na_vals=remove_na_vals,
                     description='roll_the_dice')

    def cry_with_the_wolves(self, values_a, values_b, tie_break='random', name=None, remove_na_vals=True):
        """
        Handles data conflicts by choosing the most common value. Note that when only two
        columns are being fused, matching values will be kept but non-matching value will
        must be handled with a tie-breaking strategy.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str/function tie_break: A conflict resolution function to be used to break ties. choose_random be default.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :return: None
        """
        self.resolve(vote, values_a, values_b, params=(process_tie_break(tie_break),), name=name,
                     remove_na_vals=remove_na_vals, description='cry_with_the_wolves')

    def pass_it_on(self, values_a, values_b, kind='set', name=None, remove_na_vals=True):
        """
        Data conflicts are passed on to the user. Instead of handling conflicts, all conflicting values
        are kept as a collection of values (default is a Set of values).

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str kind: The type of collection to be returned.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :return: None
        """
        self.resolve(group, values_a, values_b, params=(kind,), name=name, remove_na_vals=remove_na_vals,
                     description='pass_it_on')

    def meet_in_the_middle(self, values_a, values_b, metric, name=None, remove_na_vals=True):
        """
        Conflicting values are aggregated. Requires input values to be numeric. Note that if ``remove_na_vals``
        is False, missing data will result in np.nan value. By default, nan values are discarded during
        conflict resolution.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param metric: The aggregation metric to be used. One of 'sum', 'mean', 'stdev', 'var'.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :return: None
        """
        # Metrics Available 2017-08-01: sum, mean, stdev, var
        self.resolve(aggregate, values_a, values_b, params=(metric,), name=name, remove_na_vals=remove_na_vals,
                     description='meet_in_the_middle')

    def keep_up_to_date(self, values_a, values_b, dates_a, dates_b, tie_break='random',
                        name=None, remove_na_vals=True, remove_na_meta=True):
        """
        Keeps the most recent value. Values in values_a and values_b will be matched
        (in order) to dates in dates_a and dates_b. However, note that values and
        dates may both be "generalized" if there isn't a one-to-one correspondance between
        value columns and date columns. For example, if the user calls
        ``keep_up_to_date(['v1', 'v2'], 'v3', 'd1', 'd2')``, dates from ``d1`` would be applied
        to values from both ``c1`` and ``c2``; likewise, if ``keep_up_to_date('v1', 'v2', ['d1', 'd2'], 'd3')``
        was called, values from ``v1`` would be considered twice, associated with a date from both ``d1`` and ``d2``.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str/list dates_a: Column names for dates in df_a.
        :param str/list dates_b: Column names for dates in df_b.
        :param str/function tie_break: A conflict resolution function to be used to break ties.
        Default is choose_random.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :param bool remove_na_meta: If True, value/metadata pairs will be removed if the date is missing (i.e. np.nan).
        :return: None
        """
        self.resolve(choose_metadata_max, values_a, values_b, meta_a=dates_a, meta_b=dates_b,
                     name=name, remove_na_vals=remove_na_vals, remove_na_meta=remove_na_meta,
                     params=(process_tie_break(tie_break),), description='keep_up_to_date')

    def resolve(self, fun, values_a, values_b, meta_a=None, meta_b=None, name=None,
                transform_vals=None, transform_meta=None, static_meta=False, remove_na_vals=True,
                remove_na_meta=None, params=None, description=None, handler_override=None, **kwargs):
        """
        Queue a conflict resolution job for later computation. Conflict resolution job metadata
        is automatically stored in self.resolution_queue.

        :param function fun: A conflict resolution function.
        :param str/list values_a: Column names from df_a containing values to be resolved.
        :param str/list  values_b: Column names from df_b containing values to be resolved.
        :param str/list  meta_a: Column names from df_a containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_a will become the metadata value for all values from df_a.
        :param str/list  meta_b: Column names from df_b containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_b will become the metadata value for all values from df_b.
        :param str name: The name of the resolved column.
        :param function transform_vals: An optional pre-processing function to be applied to values.
        :param function transform_meta: An optional pre-processing function to be applied to metadata values.
        :param bool static_meta: If True, meta_a and meta_b values will be used as metadata values.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :param bool remove_na_meta: If True, value/metadata pairs will be removed if metadata is missing (i.e. np.nan).
        :param tuple params: Extra parameters used by the conflict resolution function.
        :param str description: A description string for use in logging, e.g. 'cry_with_the_wolves'.
        :param function handler_override: If specified, this function will be used to handle this job. If None,
        defaults to self._do_resolve.
        :param kwargs: Optional keyword arguments.
        :return: A dictionary of metadata values.
        """
        # TODO: Add transformation function for value -> metadata mapping.
        # TODO: Optionally specify trusted columns.
        # NOTE: Optionally, could provide value columns as metadata columns (i.e. twice) and use transform_meta.
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.

        if isinstance(remove_na_meta, bool):
            na_params = [remove_na_vals, remove_na_meta]
        else:
            na_params = [remove_na_vals]

        if params is None:
            all_params = tuple(na_params)
        elif isinstance(params, list):
            all_params = tuple(params + na_params)
        else:
            all_params = tuple(listify(params) + na_params)

        if handler_override is not None:
            handler = handler_override
        else:
            handler = self._do_resolve

        # Store metadata
        job = {
            'fun': fun,
            'values_a': values_a,
            'values_b': values_b,
            'meta_a': meta_a,
            'meta_b': meta_b,
            'transform_vals': transform_vals,
            'transform_meta': transform_meta,
            'static_meta': static_meta,
            'params': all_params,
            'name': name,
            'description': description,
            'handler': handler,
            'kwargs': kwargs
        }
        self.resolution_queue.append(job)
        return job

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None,
                                transform_vals=None, transform_meta=None, static_meta=False, **kwargs):
        """
        Formats data for conflict resolution. Output is a pandas.Series of nested tuples. _make_resolution_series
        is overriden by FuseLinks and FuseDuplicates. No implementation is provided in FuseCore.

        :param function fun: A conflict resolution function.
        :param str/list values_a: Column names from df_a containing values to be resolved.
        :param str/list  values_b: Column names from df_b containing values to be resolved.
        :param str/list  meta_a: Column names from df_a containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_a will become the metadata value for all values from df_a.
        :param str/list  meta_b: Column names from df_b containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_b will become the metadata value for all values from df_b.
        :param function transform_vals: An optional pre-processing function to be applied to values.
        :param function transform_meta: An optional pre-processing function to be applied to metadata values.
        :param bool static_meta: If True, meta_a and meta_b values will be used as metadata values.
        :param kwargs: Optional keyword arguments.
        :return: A pandas.Series.
        """
        # No implementation provided.
        # Override in subclass.
        return pd.Series()

    def _fusion_init(self, vectors, df_a, df_b, predictions, sep):
        """
        A pre-fusion initialization routine to store the data inputs for access during
        the conflict resolution / data fusion process.

        :param pandas.DataFrame vectors: Multi-indexed comparison vectors i.e. produced by recordlinkage.Compare.
        :param pandas.DataFrame df_a: The original first data frame.
        :param pandas.DataFrame df_b: The original second data frame.
        :param pandas.Series predictions: A pandas.Series of True/False classifications.
        :param str sep: A string separator to be used in resolving column naming conflicts.
        :return: None
        """
        # Comparison vectors
        self.vectors = vectors

        # Comparison / candidate link index
        self.index = vectors.index.to_frame()

        # Prediction vector
        self.predictions = predictions

        # Original data
        self.df_a = df_a
        self.df_b = df_b

        # String separator to be used when resolving duplicate names
        self._sep = sep

        # If custom names are used for index levels
        self._index_level_0 = self.index.columns[0]
        self._index_level_1 = self.index.columns[1]

        if self.predictions is not None:
            # Turn predictions into desired formats
            pred_index = self.predictions.index
            pred_list = list(self.predictions)

            # Update vectors and indices
            self.vectors = self.vectors.iloc[pred_list]
            self.index = self.vectors.index.to_frame()

    def _fusion_preprocess(self):
        """
        Subclass specific pre-fusion computation. Not implemented in FuseCore.

        :return: None
        """
        pass

    def _resolve_job_names(self, sep):
        """
        Resolves conflicts among conflict resolution job column names in self.resolution_queue.

        :param str sep: A separator string.
        :return: None
        """
        for job in self.resolution_queue:
            if job['name'] is not None:
                if job['name'] not in self._bases_taken:
                    if job['name'] not in self._names_taken:
                        self._bases_taken.append(job['name'])
                        self._names_taken.append(job['name'])
                else:
                    i = 1
                    while True:
                        name = str(job['name']) + sep + str(i)
                        if name in self._names_taken:
                            if i > 1000:
                                raise RuntimeError('Data fusion hung while attempting to _resolve column names.'
                                                   '1000 name suffixes were attempted.'
                                                   'Check for excessive name conflicts.')
                            else:
                                i += 1
                        else:
                            self._names_taken.append(name)
                            job['name'] = name
                            break

    def _handle_job(self, job):
        """
        Passes on a conflict resolution job to the appropriate handling function,
        as specified by job['handler']. Also logs activity for user.

        :param dict job: A dictionary of conflict resolution job metadata.
        :return: pandas.Series containing resolved/canonical values.
        """
        t1 = datetime.datetime.now()

        name = 'unnamed column' if job['name'] is None else str(job['name'])

        rl_logging.info(
            str(datetime.datetime.now())[:-7]
            + ':' + ' started resolving to '
            + name
            + ' (' + str(job['description']) + ')'
        )
        data = job['handler'](job)

        rl_logging.info(
            '\033[92m'
            + str(datetime.datetime.now())[:-7] + ':'
            + ' finished '
            + name
            + ' (time elapsed: '
            + str(datetime.datetime.now() - t1) + ')'
            + '\033[0m'
        )

        return data

    def _do_resolve(self, job):
        """
        Perform conflict resolution for a queued job, by pre-processing data (_make_resolution_series) and performing
        the resolution (i.e. by applying a conflict resolution function).

        :param dict job: A dictionary of conflict resolution job metadata.
        :return: pandas.Series containing resolved/canonical values.
        """

        data = self._make_resolution_series(
            job['values_a'],
            job['values_b'],
            meta_a=job['meta_a'],
            meta_b=job['meta_b'],
            transform_vals=job['transform_vals'],
            transform_meta=job['transform_meta'],
            static_meta=job['static_meta']
        )

        data = data.apply(job['fun'], args=job['params'])

        if job['name'] is not None:
            data = data.rename(job['name'])

        return data

    def fuse(self, vectors, df_a, df_b, predictions=None, n_cores=mp.cpu_count(), sep='_'):
        """
        Perform conflict resolution and data fusion for given data, using accumulated conflict resolution metadata.

        :param pandas.DataFrame vectors: Multi-indexed comparison vectors i.e. produced by recordlinkage.Compare.
        :param pandas.DataFrame df_a: The original first data frame.
        :param pandas.DataFrame df_b: The original second data frame.
        :param pandas.Series predictions: A pandas.Series of True/False classifications.
        :param n_cores: The number of cores to be used for processing. Defaults to all cores (mp.cpu_count()).
        :param str sep: A string separator to be used in resolving column naming conflicts.
        :return: A pandas.DataFrame with resolved/fused data.
        """

        if not isinstance(predictions, (type(None), pd.Series)):
            raise ValueError('Predictions must be a pandas Series.')

        # Save references to input data.
        self._fusion_init(vectors, df_a, df_b, predictions, sep)

        # Subclass-specific setup (e.g. applying refinements or detecting clusters).
        self._fusion_preprocess()

        # Resolve naming conflicts, if any.
        self._resolve_job_names(self._sep)

        # Compute resolved values for output.
        with mp.Pool(n_cores) as p:
            fused = p.map(self._handle_job, self.resolution_queue)

        return pd.concat(fused, axis=1).set_index(self.index.index)


class FuseDuplicates(FuseCore):
    def __init__(self, method=''):
        """
        ``FuseDuplicates`` is initialized without data. The initialized
        object is populated by metadata describing a series of data resolutions,
        which are executed when ``.fuse()`` is called.

        :param method: A cluster-detection algorithm. None are currently implemented.
        """
        super().__init__()
        self.method = method

    def _find_clusters(self, method):
        pass

    def _fusion_preprocess(self):
        pass

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                                transform_meta=None, static_meta=False, **kwargs):
        pass


class FuseLinks(FuseCore):
    def __init__(self):
        """
        ``FuseLinks`` is initialized without data. The initialized
        object is populated by metadata describing a series of data resolutions,
        which are executed when ``.fuse()`` is called.
        """
        super().__init__()

    def _get_df_a_col(self, name):
        """
        Returns a data from a column in df_a, corresponding to the first level of the candidate link multi-index.

        :param str name: Column name.
        :return: A ``pandas.Series``.
        """
        return self.df_a[name].loc[list(self.index[self._index_level_0])]

    def _get_df_b_col(self, name):
        """
        Returns a data from a column in df_b, corresponding to the second level of the candidate link multi-index.

        :param str name: Column name.
        :return: A ``pandas.Series``.
        """
        return self.df_b[name].loc[list(self.index[self._index_level_1])]

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                                transform_meta=None, static_meta=False, **kwargs):
        """
        Formats data for conflict resolution. Output is a pandas.Series of nested tuples.

        :param function fun: A conflict resolution function.
        :param str/list values_a: Column names from df_a containing values to be resolved.
        :param str/list  values_b: Column names from df_b containing values to be resolved.
        :param str/list  meta_a: Column names from df_a containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_a will become the metadata value for all values from df_a.
        :param str/list  meta_b: Column names from df_b containing metadata values to be used in conflict resolution.
        Optionally, if static_meta is True, meta_b will become the metadata value for all values from df_b.
        :param function transform_vals: An optional pre-processing function to be applied to values.
        :param function transform_meta: An optional pre-processing function to be applied to metadata values.
        :param bool static_meta: If True, meta_a and meta_b values will be used as metadata values.
        :param kwargs: Optional keyword arguments.
        :return: A pandas.Series.
        """

        if self.df_a is None:
            raise AssertionError('df_a is None')

        if self.df_b is None:
            raise AssertionError('df_b is None')

        if transform_vals is not None and callable(transform_vals) is not True:
            raise ValueError('transform_vals must be callable.')

        if transform_meta is not None and callable(transform_meta) is not True:
            raise ValueError('transform_meta must be callable.')

        # Listify value inputs
        values_a = listify(values_a)
        values_b = listify(values_b)

        # Listify and validate metadata inputs
        if (meta_a is None and meta_b is not None) or (meta_b is None and meta_a is not None):
            raise AssertionError('Metadata was given for one Data Frame but not the other.')

        if meta_a is None and meta_b is None:
            use_meta = False
        elif static_meta is False:
            use_meta = True
            meta_a = listify(meta_a)
            meta_b = listify(meta_b)
        else:
            use_meta = True

        # Check value / metadata column correspondence
        if use_meta is True and static_meta is False:

            if len(values_a) < len(meta_a):
                generalize_values_a = True
                generalize_meta_a = False
                rl_logging.warn('Generalizing values. There are fewer columns in values_a than in meta_a. '
                                'Values in first column of values_a will be generalized to values in meta_a.')
            elif len(values_a) > len(meta_a):
                generalize_values_a = False
                generalize_meta_a = True
                rl_logging.warn('Generalizing metadata. There are fewer columns in meta_a than in values_a. '
                                'Values in first column of meta_a will be generalized to values in values_a.')
            else:
                generalize_values_a = False
                generalize_meta_a = False

            if len(values_b) < len(meta_b):
                generalize_values_b = True
                generalize_meta_b = False
                rl_logging.warn('Generalizing values. There are fewer columns in values_b than in meta_b. '
                                'Values in first column of values_b will be generalized to values in meta_b.')
            elif len(values_b) > len(meta_b):
                generalize_values_b = False
                generalize_meta_b = True
                rl_logging.warn('Generalizing metadata. There are fewer columns in meta_b than in values_b. '
                                'Values in first column of meta_b will be generalized to values in values_b.')
            else:
                generalize_values_b = False
                generalize_meta_b = False
        else:
            generalize_values_a = None
            generalize_meta_a = None
            generalize_values_b = None
            generalize_meta_b = None

        # Make list of data series
        data_a = []
        if generalize_values_a is True:
            for _ in range(len(meta_a)):
                data_a.append(
                    self._get_df_a_col(values_a[0])
                )
        else:
            for name in values_a:
                data_a.append(
                    self._get_df_a_col(name)
                )

        data_b = []

        if generalize_values_b is True:
            for _ in range(len(meta_b)):
                data_b.append(
                    self._get_df_b_col(values_b[0])
                )
        else:
            for name in values_b:
                data_b.append(
                    self._get_df_b_col(name)
                )

        # Combine data
        value_data = data_a
        value_data.extend(data_b)

        # Apply transformation if function is provided
        if transform_vals is not None:
            value_data = [s.apply(transform_vals) for s in value_data]

        # Zip data
        value_data = zip(*value_data)

        # Make list of metadata series
        if use_meta is True:

            metadata_a = []

            if static_meta is True:
                for _ in range(len(values_a)):
                    metadata_a.append(
                        pd.Series([meta_a for _ in range(len(self.index))], index=self.index[self._index_level_0])
                    )
            elif generalize_meta_a is True:
                for _ in range(len(values_a)):
                    metadata_a.append(
                        self._get_df_a_col(meta_a[0])
                    )
            else:
                for name in meta_a:
                    metadata_a.append(
                        self._get_df_a_col(name)
                    )

            metadata_b = []

            if static_meta is True:
                for _ in range(len(values_b)):
                    metadata_b.append(
                        pd.Series([meta_b for _ in range(len(self.index))], index=self.index[self._index_level_1])
                    )
            elif generalize_meta_b is True:
                for _ in range(len(values_b)):
                    metadata_b.append(
                        self._get_df_b_col(meta_b[0])
                    )
            else:
                for name in meta_b:
                    metadata_b.append(
                        self._get_df_b_col(name)
                    )

            # Combine metadata
            metadata = metadata_a
            metadata.extend(metadata_b)

            # Apply transformation if function is provided
            if transform_meta is not None:
                metadata = [s.apply(transform_meta) for s in metadata]

            # Zip metadata
            metadata = zip(*metadata)
        else:
            metadata = None

        if use_meta is True:
            output = pd.Series(list(zip(value_data, metadata)))
        else:
            output = pd.Series(list(zip(value_data)))

        return output

    def _do_keep(self, job):
        """
        Handles a conflict resolution job created by FuseLinks.keep_original, where data is included
        unaltered from one data source. Using this handling method bypasses the overhead
        of _get_resolution_series and conflict handling functions, which are unnecessary
        in this case.

        :param dict job: A dictionary of conflict resolution job metadata.
        :return: pandas.Series containing resolved/canonical values.
        """

        vals_a = listify(job['values_a'])
        vals_b = listify(job['values_b'])

        source_a = len(vals_a) == 1
        source_b = len(vals_b) == 1
        # enforce xor condition: There is one value in values_a or values_b but not both
        if source_a == source_b:
            raise AssertionError(
                '_do_keep only operates on a single column from a single source.'
                'Was given job["values_a"] = {}, and job["values_b"] = {}'.format(
                    vals_a,
                    vals_b,
                ))

        if source_a:
            data = self._get_df_a_col(vals_a[0])
        else:
            data = self._get_df_b_col(vals_b[0])

        if callable(job['transform_vals']):
            data = data.apply(job['transform_vals'])

        if job['name'] is not None:
            data = data.rename(job['name'])

        data = data.reset_index(drop=True)

        return data

    def keep_original(self, columns_a, columns_b, suffix_a=None, suffix_b=None, sep='_'):
        """
        Specifies columns from df_a and df_b which should be included in the fused output, but
        which do not require conflict resolution. This methods queues a job in self.resolution_queue
        using the ``identity`` conflict resolution function.

        :param str/list columns_a: A list of column names to be included from df_a.
        :param str/list columns_b: A list of column names to be included from df_b.
        :param str suffix_a: An optional suffix to be applied to the name of all columns kept from df_a.
        :param str suffix_b: An optional suffix to be applied to the names of all columns kept from df_b.
        :param str sep: The separator that should be used when resolving column name conflicts (e.g.
        with ``sep='_'``, ``taken_name`` becomes ``taken_name_1``..
        :return: None
        """

        columns_a = listify(columns_a)
        columns_b = listify(columns_b)

        for col in columns_a:
            if suffix_a is None:
                self.resolve(identity, [col], [], name=col, handler_override=self._do_keep)
            else:
                self.resolve(identity, [col], [], name=col + sep + str(suffix_a), handler_override=self._do_keep)

        for col in columns_b:
            if suffix_b is None:
                self.resolve(identity, [], [col], name=col, handler_override=self._do_keep)
            else:
                self.resolve(identity, [], [col], name=col + sep + str(suffix_b), handler_override=self._do_keep)

    # FuseLinks Conflict Resolution Realizations

    def trust_your_friends(self, values_a, values_b, trusted, tie_break_trusted='random',
                           tie_break_untrusted='random', name=None, remove_na_vals=True):
        """
        Handles data conflicts by keeping data from a trusted source.

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str trusted: A preferred data source. 'a' for df_a or 'b' for df_b.
        :param str/function tie_break_trusted: A conflict resolution function to be to break ties between trusted values.
        :param str/function tie_break_untrusted: A conflict resolution function to be to break ties between trusted values.
        :param str name: The name of the resolved column.
        :param bool remove_na_vals: If True, value/metadata pairs will be removed if the value is missing (i.e. np.nan).
        :return: None
        """
        self.resolve(choose_trusted, values_a, values_b, meta_a='a', meta_b='b', name=name, static_meta=True,
                     params=(trusted, process_tie_break(tie_break_trusted), process_tie_break(tie_break_untrusted)),
                     remove_na_vals=remove_na_vals, remove_na_meta=False, description='trust_your_friends')
