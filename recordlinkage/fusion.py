from __future__ import division
from __future__ import unicode_literals

import inspect
import datetime
import warnings
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from six import add_metaclass, string_types, text_type

import numpy as np
import pandas as pd

from recordlinkage import logging as rl_logging
from recordlinkage.utils import listify, multi_index_to_frame
from recordlinkage.algorithms.conflict_resolution import (
    choose_trusted, choose_first, choose_last, choose_max, choose_min,
    choose_shortest_tie_break, choose_longest_tie_break, choose_metadata_max,
    choose_metadata_min, choose_random, aggregate, group, no_gossip, vote,
    nullify)


class SkipNull(object):
    """
    This object is used like a function decorator for ignoring missing
    values when applying a transformation function. An object is
    used in place of a function because local functions cannot
    be pickled, causing errors during multiprocessing.
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        if pd.isnull(x):
            return x
        else:
            return self.f(x)


@add_metaclass(ABCMeta)
class FusionHandler:
    def __init__(self, fuse):
        self.fuse = fuse

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ResolveHandler(FusionHandler):
    def __call__(self, job):
        """

        Perform conflict resolution for a queued job, by pre-
        processing data (_make_resolution_series) and performing the
        resolution (i.e. by applying a conflict resolution function).

        Parameters
        ----------
        fuse : FuseCore
            The originating FuseCore object.
        job : dict
            A dictionary of conflict resolution job metadata.

        Returns
        -------
        pandas.Series
            Series of resolved/canonical values.

        """

        data = self.fuse._make_resolution_series(
            job['values_a'],
            job['values_b'],
            meta_a=job['meta_a'],
            meta_b=job['meta_b'],
            transform_vals=job['transform_vals'],
            transform_meta=job['transform_meta'],
            static_meta=job['static_meta'])

        data = data.apply(job['fun'], args=job['params'])

        if job['name'] is not None:
            data = data.rename(job['name'])

        return data


class KeepHandler(FusionHandler):
    def __call__(self, job):
        """

        Handles a conflict resolution job created by
        FuseLinks.keep_original, where data is included unaltered from
        one data source. Using this handling method bypasses the
        overhead of _get_resolution_series and conflict handling
        functions, which are unnecessary in this case. Note that not
        all options supported for _do_resolve are supported for
        _do_keep (e.g. transform_vals).

        Parameters
        ----------
        fuse : FuseCore
            The originating FuseCore object.
        job : dict
            A dictionary of conflict resolution job metadata.

        Returns
        -------
        pandas.Series
            pandas.Series containing resolved/canonical values.

        """

        vals_a = listify(job['values_a'])
        vals_b = listify(job['values_b'])

        source_a = len(vals_a) == 1
        source_b = len(vals_b) == 1
        # enforce xor condition: There is one value in values_a or values_b
        # but not both
        if source_a == source_b:
            raise AssertionError(
                '_do_keep only operates on a single column from a single'
                'source. Was given job["values_a"] = {}, and job["values_b"]'
                ' = {}'.format(vals_a, vals_b))

        if source_a:
            data = self.fuse._get_df_a_col(vals_a[0])
        else:
            data = self.fuse._get_df_b_col(vals_b[0])

        if job['name'] is not None:
            data = data.rename(job['name'])

        data = data.reset_index(drop=True)

        return data


def handle_job(job):
    """
    Passes on a conflict resolution job to the appropriate handling function,
    as specified by job['handler']. Also logs activity for user.

    Parameters
    ----------
    fuse : FuseCore
        The originating FuseCore object.
    job : dict
        A dictionary of conflict resolution job metadata.

    Returns
    -------
    pandas.Series
        Series of resolved/canonical values.

    """
    t1 = datetime.datetime.now()

    name = 'unnamed column' if job['name'] is None else str(job['name'])

    rl_logging.info(
        'started resolving to ' + name + ' (' + str(job['description']) + ')')
    data = job['handler'](job)

    rl_logging.info('\033[92m' + 'finished ' + name + ' (time elapsed: ' +
                    str(datetime.datetime.now() - t1) + ')' + '\033[0m')

    return data


def process_tie_break(tie_break):
    """
    Handles string/function -> function mapping for tie break options.

    A "tie-breaking function" is a special case of conflict-resolution
    functions which must:
        * Have a signature like tie_break_fun(x, remove_na_vals)
        * Use values only - no metadata values
        * Not require tiebreaking

    Parameters
    ----------
    tie_break : str/function
        A conflict resolution function to be used in the case of a
        tie. May be a conflict resolution function or a string (one
        of 'random', 'trust_a', 'trust_b', 'first', 'last', 'min',
        'max', 'shortest', 'longest', or 'null').

    Returns
    -------
    Function
        A tie break function

    """
    tie_break_fun = None
    if tie_break is None:
        raise ValueError('given None as tie_break strategy')
    elif callable(tie_break):
        tie_break_fun = tie_break
    elif type(tie_break) in [text_type] + list(string_types):
        if tie_break == 'random':
            tie_break_fun = choose_random
        elif tie_break == 'trust_a' or tie_break == 'first':
            tie_break_fun = choose_first
        elif tie_break == 'trust_b' or tie_break == 'last':
            tie_break_fun = choose_last
        elif tie_break == 'min':
            tie_break_fun = choose_min
        elif tie_break == 'max':
            tie_break_fun = choose_max
        elif tie_break == 'shortest':
            tie_break_fun = choose_shortest_tie_break
        elif tie_break == 'longest':
            tie_break_fun = choose_longest_tie_break
        elif tie_break == 'null' or tie_break == 'nullify':
            tie_break_fun = nullify
        else:
            raise ValueError(
                'Invalid tie_break strategy: {}. Must be one of '
                'random, trust_a, trust_b, first, last, min, max, '
                'shortest, longest, or nullify.'.format(tie_break))
    else:
        raise ValueError(
            'tie_break must be a string or a function. Got {} ({}).'.format(
                tie_break, type(tie_break)))
    return tie_break_fun


@add_metaclass(ABCMeta)
class FuseCore:
    def __init__(self):
        """

        ``FuseCore`` and its subclasses are initialized without data.
        The initialized object is populated by metadata describing a
        series of data resolutions, which are executed when
        ``.fuse()`` is called.

        """
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

    def no_gossiping(self, values_a, values_b, name=None):
        """No gossiping strategy to fuse data.

        Handles data conflicts by keeping values agree upon by both
        data sources, and returning np.nan for conflicting or missing
        values.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        name : str
            The name of the resulting resolved column.

        Returns
        -------
        None
            Queues a conflict resolution job for later computation, to
            be completed when ``fuse`` method is called.
        """
        self.resolve(
            no_gossip,
            values_a,
            values_b,
            name=name,
            remove_na_vals=False,
            description='no_gossiping')

    def roll_the_dice(self, values_a, values_b, name=None,
                      remove_na_vals=True):
        """Choose a random non-missing value.

        Handles data conflicts by choosing a random non-missing value.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).
            If False, np.nan values may also be chosen.

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.
        """
        self.resolve(
            choose_random,
            values_a,
            values_b,
            name=name,
            remove_na_vals=remove_na_vals,
            description='roll_the_dice')

    def cry_with_the_wolves(self,
                            values_a,
                            values_b,
                            tie_break='random',
                            name=None,
                            remove_na_vals=True):
        """Choose the most common value.

        Handles data conflicts by choosing the most common value. Note
        that when only two columns are being fused, matching values
        will be kept but non-matching value will must be handled with
        a tie-breaking strategy.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        tie_break : str/function
            A conflict resolution function to be used to break ties.
            Uses choose_random by default.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        self.resolve(
            vote,
            values_a,
            values_b,
            tie_break=process_tie_break(tie_break),
            name=name,
            remove_na_vals=remove_na_vals,
            description='cry_with_the_wolves')

    def pass_it_on(self,
                   values_a,
                   values_b,
                   kind='list',
                   name=None,
                   remove_na_vals=True):
        """Return values.

        Data conflicts are passed on to the user. Instead of handling
        conflicts, all conflicting values are kept as a collection of
        values (default is a List of values).

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        kind : str
            The type of collection to be returned. One of 'list',
            'set', or 'tuple'.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        self.resolve(
            group,
            values_a,
            values_b,
            params=kind,
            name=name,
            remove_na_vals=remove_na_vals,
            description='pass_it_on')

    def meet_in_the_middle(self,
                           values_a,
                           values_b,
                           metric,
                           name=None,
                           remove_na_vals=True):
        """Meet in the middle approach.

        Conflicting values are aggregated. Input values must be
        numeric. Note that if ``remove_na_vals`` is False, missing
        data will result in np.nan value. By default, nan values are
        discarded during conflict resolution.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        metric : str
            The aggregation metric to be used. One of 'sum', 'mean',
            'stdev', 'var'.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the
            value is missing (i.e. np.nan).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        self.resolve(
            aggregate,
            values_a,
            values_b,
            params=metric,
            name=name,
            remove_na_vals=remove_na_vals,
            description='meet_in_the_middle')

    def keep_up_to_date(self,
                        values_a,
                        values_b,
                        dates_a,
                        dates_b,
                        tie_break='random',
                        name=None,
                        remove_na_vals=True,
                        remove_na_dates=True):
        """Keep most recent.

        Keeps the most recent value. Values in values_a and values_b
        will be matched to corresponding dates from dates_a and
        dates_b. However, note that values and dates may both be
        "generalized" if there isn't a one-to-one correspondance
        between value columns and date columns. For example, if the
        user calls ``keep_up_to_date(['v1', 'v2'], 'v3', 'd1',
        'd2')``, dates from ``d1`` would be applied to values from
        both ``v1`` and ``v2``; likewise, if ``keep_up_to_date('v1',
        'v2', ['d1', 'd2'], 'd3')`` was called, values from ``v1``
        would be considered twice, associated with a date from both
        ``d1`` and ``d2``.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        dates_a : str/list
            Column names for dates in df_a.
        dates_b : str/list
            Column names for dates in df_b.
        tie_break : str/function
            A conflict resolution function to be used to break ties.
            Default is choose_random.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).
        remove_na_dates : bool
            If True, value/metadata pairs will be removed if the date
            is missing (i.e. np.nan).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        self.resolve(
            choose_metadata_max,
            values_a,
            values_b,
            meta_a=dates_a,
            meta_b=dates_b,
            name=name,
            remove_na_vals=remove_na_vals,
            remove_na_meta=remove_na_dates,
            tie_break=process_tie_break(tie_break),
            description='keep_up_to_date')

    def choose_by_scored_value(self,
                               values_a,
                               values_b,
                               func,
                               tie_break='random',
                               apply_to_null=False,
                               name=None,
                               remove_na_vals=True,
                               choose_max_value=True):
        """Choose by scored value.

        Chooses the value which is given the highest (or lowest)
        numeric score given by a user-specified function. The scoring
        function may recognize specific features, compute taxonomic
        depth, handle unusual data types, etc.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        func : function
            A scoring function, which takes a single data value and
            returns a numeric score. Note that if apply_to_null is
            True, func must handle missing (np.nan) values.
        tie_break : str/function
            A conflict resolution function to be used to break ties.
            Default is choose_random.
        apply_to_null : bool
            If False, missing values (np.nan) will not be scored by
            the user-specified funciton. If True, any missing values
            much be handled inside the user-specified function.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, missing values will be ignored.
        choose_max_value : bool
            If True, the largest value will be chosen. If False, the
            smallest value will be chosen.

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        if choose_max_value:
            resolution_function = choose_metadata_max
        else:
            resolution_function = choose_metadata_min

        self.resolve(
            resolution_function,
            values_a,
            values_b,
            meta_a=values_a,
            meta_b=values_b,
            name=name,
            remove_na_vals=remove_na_vals,
            remove_na_meta=remove_na_vals,
            tie_break=process_tie_break(tie_break),
            description='choose_by_scored_value',
            transform_meta=func,
            transform_null=apply_to_null)

    def choose_by_scored_metadata(self,
                                  values_a,
                                  values_b,
                                  meta_a,
                                  meta_b,
                                  func,
                                  tie_break='random',
                                  apply_to_null=False,
                                  name=None,
                                  remove_na_vals=True,
                                  remove_na_meta=True,
                                  choose_max_value=True):
        """Choose by score metadata.

        Chooses the value with a corresponding metadata value which is
        given the highest (or lowest) numeric score by a user-
        specified function. The scoring function may recognize
        specific features, compute taxonomic depth, handle unusual
        datatypes, etc.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        meta_a : str/list
            Column name(s) from df_a containing metadata values to be
            used to choose values.
        meta_b : str/list
            Column name(s) from df_b containing metadata values to be
            used to choose values.
        func : function
            A scoring function, which takes a single metadata data
            value and returns a numeric score. Note that if
            apply_to_null is True, func must handle missing (np.nan)
            values.
        tie_break : str/function
            A conflict resolution function to be used to break ties.
            Default is choose_random.
        apply_to_null : bool
            If False, missing values (np.nan) will not be scored by
            the user-specified funciton. If True, any missing values
            much be handled inside the user-specified function.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).
        remove_na_meta : bool
            If True, value/metadata pairs will be removed if metadata
            is missing (i.e. np.nan).
        choose_max_value : bool
            If True, the value with the largest metadata value will be
            chosen. If False, the value with the smallest metadata
            value will be chosen.

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        if choose_max_value:
            resolution_function = choose_metadata_max
        else:
            resolution_function = choose_metadata_min

        self.resolve(
            resolution_function,
            values_a,
            values_b,
            meta_a=meta_a,
            meta_b=meta_b,
            name=name,
            remove_na_vals=remove_na_vals,
            remove_na_meta=remove_na_meta,
            tie_break=process_tie_break(tie_break),
            description='choose_by_scored_metadata',
            transform_meta=func,
            transform_null=apply_to_null)

    def resolve(self,
                fun,
                values_a,
                values_b,
                meta_a=None,
                meta_b=None,
                name=None,
                tie_break=None,
                transform_vals=None,
                transform_meta=None,
                transform_null=False,
                static_meta=False,
                remove_na_vals=True,
                remove_na_meta=None,
                params=None,
                description=None,
                handler_override=None,
                **kwargs):
        """General method to queue jobs.

        A general-purpose method to queue a conflict resolution job
        for later computation. Conflict resolution job metadata is
        stored in self.resolution_queue.

        Parameters
        ----------
        fun : function
            A conflict resolution function.
        values_a : str/list
            Column name(s) from df_a containing values to be resolved.
        values_b :
            Column name(s) from df_b containing values to be resolved.
        meta_a :
            Column name(s) from df_a containing metadata values to be
            used to choose values. Optionally, if static_meta is True,
            meta_a will become the metadata value for all values from
            df_a.
        meta_b :
            Column name(s) from df_b containing metadata values to be
            used to choose values. Optionally, if static_meta is True,
            meta_b will become the metadata value for all values from
            df_b.
        name : str
            The name of the resulting resolved column.
        transform_vals : function
            An optional pre-processing function to be applied to values.
        transform_meta : function
            An optional pre-processing function to be applied to
            metadata values.
        transform_null : bool
            If True, transform_vals/transform_meta will be called on
            missing values. If False (default) missing values are
            skipped automatically.
        static_meta : bool
            If True, the user-specified values of meta_a and meta_b
            will be used as metadata for all values. Useful if the
            user wishes to preserve information about the source that
            a value came from, such as the original dataframe or
            column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).
        remove_na_meta : bool
            If True, value/metadata pairs will be removed if metadata
            is missing (i.e. np.nan).
        params : tuple/list
            Extra arguments used by the conflict resolution function
            (e.g. for the ``metric`` parameter of the ``aggregate``
            function).
        description : str
            A description string for use in logging, e.g.
            'cry_with_the_wolves'.
        handler_override : function
            If specified, this function will be used to handle this
            job. If None,defaults to do_resolve.

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """

        if isinstance(remove_na_meta, bool):
            na_params = [remove_na_vals, remove_na_meta]
        else:
            na_params = [remove_na_vals]

        if params is None:
            all_params = tuple(listify(tie_break) + na_params)
        elif isinstance(params, list):
            all_params = tuple(params + listify(tie_break) + na_params)
        else:
            all_params = tuple(
                listify(params) + listify(tie_break) + na_params)

        if fun is not None:
            argspec = inspect.getargspec(fun)[0]

            # Check that the given arguments are appropriate for the specified
            # conflict resolution function.

            param_pairs = [(remove_na_vals, 'remove_na_vals'),
                           (remove_na_meta, 'remove_na_meta'), (tie_break,
                                                                'tie_break')]
            for given, param_name in param_pairs:
                if given is None:
                    if param_name in argspec:
                        raise AssertionError(
                            'Missing argument. {} requires {}'.format(
                                fun.__name__, param_name))
                else:
                    if param_name not in argspec:
                        raise AssertionError(
                            'Incorrect arguments. {} does not take {}'.format(
                                fun.__name__, param_name))
            if len(argspec) != len(all_params) + 1:
                raise AssertionError(
                    'Incorrect arguments. The number of options specified do'
                    'not match the conflict resolution funciton signature.'
                    'Expected: ' + str(argspec) + ' got: ' +
                    str(['x'] + list(all_params)))

        if handler_override is not None:
            handler = handler_override
        else:
            handler = ResolveHandler(self)

        # Store metadata
        job = {
            'fun':
            fun,
            'values_a':
            values_a,
            'values_b':
            values_b,
            'meta_a':
            meta_a,
            'meta_b':
            meta_b,
            'transform_vals':
            transform_vals if transform_null or transform_vals is None else
            SkipNull(transform_vals),
            'transform_meta':
            transform_meta if transform_null or transform_vals is None else
            SkipNull(transform_meta),
            'static_meta':
            static_meta,
            'params':
            all_params,
            'name':
            name,
            'description':
            description,
            'handler':
            handler,
            'kwargs':
            kwargs
        }
        self.resolution_queue.append(job)

    @abstractmethod
    def _make_resolution_series(self,
                                values_a,
                                values_b,
                                meta_a=None,
                                meta_b=None,
                                transform_vals=None,
                                transform_meta=None,
                                static_meta=False,
                                **kwargs):
        """

        Formats data for conflict resolution. Output is a
        pandas.Series of nested tuples. _make_resolution_series is
        overriden by FuseLinks and FuseDuplicates. No implementation
        is provided in FuseCore.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a containing values to be resolved.
        values_b :
            Column name(s) from df_b containing values to be resolved.
        meta_a :
            Column name(s) from df_a containing metadata values to be
            used in conflict resolution. Optionally, if static_meta is
            True, meta_a will become the metadata value for all values
            from df_a.
        meta_b :
            Column name(s) from df_b containing metadata values to be
            used in conflict resolution. Optionally, if static_meta is
            True, meta_b will become the metadata value for all values
            from df_b.
        transform_vals : function
            An optional pre-processing function to be applied to
            values.
        transform_meta : function
            An optional pre-processing function to be applied to
            metadata values.
        static_meta : bool
            If True, the user-specified values of meta_a and meta_b
            will be used as metadata for all values. Useful if the
            user wishes to preserve information about the source that
            a value came from, such as the original dataframe or
            column.

        Returns
        -------
        pandas.Series
            A series of nested tuples containing values to be
            resolved, and optional metadata values. Value-only tuples
            are of form :math:`((val_1, ..., val_n), )` whereas
            value-metadata pairs are represented as
            :math:`((val_1, ..., val_n), (meta_1, ..., meta_n))` where
            val_i and meta_i are a value/metadata pair.

        """
        # No implementation provided.
        # Override in subclass.
        return NotImplemented

    def _fusion_init(self, index, df_a, df_b, predictions, sep):
        """

        A pre-fusion initialization routine to store the data inputs
        for access during the conflict resolution / data fusion
        process.

        Parameters
        ----------
        index : pandas.MultiIndex
            MultiIndex describing a set of candidate links / record
            pairs (e.g. produced by recordlinkage Indexing classes or
             a recordlinkage.Compare.vectors.index).
        df_a : pandas.DataFrame
            The original first data frame.
        df_b : pandas.DataFrame
            The original second data frame.
        predictions : pandas.Series
            A pandas.Series of True/False classifications.
        sep : str
            A string separator to be used in resolving column naming
            conflicts.

        Returns
        -------
        None

        """
        # Comparison / candidate link index. Remove names in case of name
        # collision.
        if len(set(index.names)) != len(index.names):
            self.index = multi_index_to_frame(
                index.set_names(list(range(len(index.names)))))
        else:
            self.index = multi_index_to_frame(index)

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
            pred_list = list(self.predictions)

            # Update multiindex
            self.index = self.index.loc[pred_list]

    def _resolve_job_names(self, sep):
        """
        Resolves conflicts among conflict resolution job column names
        in self.resolution_queue.

        Parameters
        ----------
        sep : str
            A separator string.

        Returns
        -------
        None

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
                                raise RuntimeError(
                                    'Data fusion hung while attempting to'
                                    '_resolve column names.'
                                    '1000 name suffixes were attempted.'
                                    'Check for excessive name conflicts.')
                            else:
                                i += 1
                        else:
                            self._names_taken.append(name)
                            job['name'] = name
                            break

    def fuse(self, index, df_a, df_b, predictions=None, njobs=1, sep='_'):
        """Fuse the data with the given links.

        Perform conflict resolution and data fusion for given data,
        using accumulated conflict resolution metadata.

        Parameters
        ----------
        index : pandas.MultiIndex
            MultiIndex describing a set of candidate links / record
            pairs (e.g. produced by recordlinkage Indexing classes or
            a recordlinkage.Compare.vectors.index).
        df_a : pandas.DataFrame
            The original first data frame.
        df_b : pandas.DataFrame
            The original second data frame.
        predictions : pandas.Series
            A pandas.Series of True/False classifications.
        njobs : int
            The number of cores to be used for processing. Defaults to
            one core. If njobs=1, multiprocessing will not be used.
        sep : str
            A string separator to be used in resolving column naming
            conflicts.

        Returns
        -------
        pandas.DataFrame
            A pandas.DataFrame with resolved/fused data.

        """

        if not isinstance(predictions, (type(None), pd.Series, type(list))):
            raise ValueError(
                'Predictions must be a pandas Series, list, or None.')

        if predictions is not None:
            if len(predictions) != len(index):
                raise AssertionError(
                    'Length of the predictions vector ({}) does not match '
                    'the length of the index ({}).'.format(
                        len(predictions), len(index)))

            if isinstance(predictions, pd.Series):
                use_predictions = predictions
            else:
                use_predictions = pd.Series(predictions)
                use_predictions.index = self.index.index

            if use_predictions.dtype == np.dtype(bool):
                pass
            elif use_predictions.dtype == np.dtype(int):
                rl_logging.warning(
                    'Expected predictions to be a boolean vector, but got'
                    'vector of integers. Coercing integer values to boolean '
                    'values.')
                use_predictions = use_predictions.apply(bool)
            else:
                raise ValueError(
                    'predictions must be a pandas.Series (or list) of '
                    'boolean values.')
        else:
            use_predictions = None

        # Save references to input data.
        self._fusion_init(index, df_a, df_b, use_predictions, sep)

        # Resolve naming conflicts, if any.
        self._resolve_job_names(self._sep)

        # Perform conflict resolution from resolution queue metadata, using
        # the appropriate number of cores (and multiprocessing if applicable.)
        if njobs <= 1:
            fused = [
                handle_job(job_data) for job_data in self.resolution_queue
            ]
        else:
            if njobs > mp.cpu_count():
                warnings.warn('njobs exceeds maximum available cores ({}). '
                              'Defaulting to maximum available.'.format(
                                  mp.cpu_count()), RuntimeWarning)
                use_n_cores = mp.cpu_count()
            else:
                use_n_cores = njobs
            # Compute resolved values for output.
            p = mp.Pool(use_n_cores)
            fused = p.map(handle_job, self.resolution_queue)
            p.close()

        return pd.concat(fused, axis=1).set_index(self.index.index)


class FuseDuplicates(FuseCore):
    def __init__(self, method=''):
        """

        ``FuseDuplicates`` is initialized without data. The
        initialized object is populated by metadata describing a
        series of data resolutions, which are executed when
        ``.fuse()`` is called.

        :param method: A cluster-detection algorithm. None are
            currently implemented.
        """
        super(FuseDuplicates, self).__init__()
        self.method = method
        warnings.warn('FuseDuplicates has not been implemented.')

    def _find_clusters(self, method):
        warnings.warn('FuseDuplicates has not been implemented.')
        return NotImplemented

    def _make_resolution_series(self,
                                values_a,
                                values_b,
                                meta_a=None,
                                meta_b=None,
                                transform_vals=None,
                                transform_meta=None,
                                static_meta=False,
                                **kwargs):
        warnings.warn('FuseDuplicates has not been implemented.')
        return NotImplemented


class FuseLinks(FuseCore):
    """

    FuseLinks turns two linked data frames into a single "fused" data
    frame, with options to handle data conflicts between columns in
    the two data frames.

    Note that FuseLinks handles conflicts between record pairs (i.e.
    candidate links). It may not be suitable for data duplication
    problems, where the user may need to fuse *clusters* of records.
    (A ``FuseDuplicates`` class which detects and fuses clusters may
    be implemented in the future.)

    """

    def __init__(self):
        """
        ``FuseLinks`` is initialized without data. The initialized
        object is populated by metadata describing a series of data
        resolutions, which are executed when ``.fuse()`` is called.
        """
        super(FuseLinks, self).__init__()

    def _get_df_a_col(self, name):
        """

        Returns a data from a column in df_a, corresponding to the
        first level of the candidate link MultiIndex.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        pandas.Series

        """
        return self.df_a[name].loc[list(self.index[self._index_level_0])]

    def _get_df_b_col(self, name):
        """

        Returns a data from a column in df_b, corresponding to the
        second level of the candidate link MultiIndex.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        pandas.Series

        """
        return self.df_b[name].loc[list(self.index[self._index_level_1])]

    def _make_resolution_series(self,
                                values_a,
                                values_b,
                                meta_a=None,
                                meta_b=None,
                                transform_vals=None,
                                transform_meta=None,
                                static_meta=False,
                                **kwargs):
        """

        Formats data for conflict resolution. Output is a
        pandas.Series of nested tuples.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a containing values to be resolved.
        values_b :
            Column name(s) from df_b containing values to be resolved.
        meta_a :
            Column name(s) from df_a containing metadata values to be
            used in conflict resolution.Optionally, if static_meta is
            True, meta_a will become the metadata value for all values
            from df_a.
        meta_b :
            Column name(s) from df_b containing metadata values to be
            used in conflict resolution.Optionally, if static_meta is
            True, meta_b will become the metadata value for all values
            from df_b.
        transform_vals : function
            An optional pre-processing function to be applied to
            values.
        transform_meta : function
            An optional pre-processing function to be applied to
            metadata values.
        static_meta : bool
            If True, the user-specified values of meta_a and meta_b
            will be used as metadata for all values. Useful if the
            user wishes to preserve information about the source that
            a value came from, such as the original dataframe or
            column.

        Returns
        -------
        A pandas.Series

        """

        if self.df_a is None:
            raise AssertionError('df_a is None')

        if self.df_b is None:
            raise AssertionError('df_b is None')

        if transform_vals is not None and not callable(transform_vals):
            raise ValueError('transform_vals must be callable.')

        if transform_meta is not None and not callable(transform_meta):
            raise ValueError('transform_meta must be callable.')

        # Listify value inputs
        values_a = listify(values_a)
        values_b = listify(values_b)

        # Listify and validate metadata inputs
        if (meta_a is None and meta_b is not None) or \
           (meta_b is None and meta_a is not None):
            raise AssertionError(
                'Metadata was given for one Data Frame but not the other.')

        if meta_a is None and meta_b is None:
            use_meta = False
        elif not static_meta:
            use_meta = True
            meta_a = listify(meta_a)
            meta_b = listify(meta_b)
        else:
            use_meta = True

        # Check value / metadata column correspondence
        if use_meta and not static_meta:

            if len(values_a) < len(meta_a):
                generalize_values_a = True
                generalize_meta_a = False
                rl_logging.warning(
                    'Generalizing values. There are fewer columns in '
                    'values_a than in meta_a. Values in first column of '
                    'values_a will be generalized to values in meta_a.')
            elif len(values_a) > len(meta_a):
                generalize_values_a = False
                generalize_meta_a = True
                rl_logging.warning(
                    'Generalizing metadata. There are fewer columns in '
                    'meta_a than in values_a. Values in first column of '
                    'meta_a will be generalized to values in values_a.')
            else:
                generalize_values_a = False
                generalize_meta_a = False

            if len(values_b) < len(meta_b):
                generalize_values_b = True
                generalize_meta_b = False
                rl_logging.warning(
                    'Generalizing values. There are fewer columns in '
                    'values_b than in meta_b. Values in first column of '
                    'values_b will be generalized to values in meta_b.')
            elif len(values_b) > len(meta_b):
                generalize_values_b = False
                generalize_meta_b = True
                rl_logging.warning(
                    'Generalizing metadata. There are fewer columns in '
                    'meta_b than in values_b. Values in first column of '
                    'meta_b will be generalized to values in values_b.')
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
        if generalize_values_a:
            for _ in range(len(meta_a)):
                data_a.append(self._get_df_a_col(values_a[0]))
        else:
            for name in values_a:
                data_a.append(self._get_df_a_col(name))

        data_b = []

        if generalize_values_b:
            for _ in range(len(meta_b)):
                data_b.append(self._get_df_b_col(values_b[0]))
        else:
            for name in values_b:
                data_b.append(self._get_df_b_col(name))

        # Combine data
        value_data = data_a
        value_data.extend(data_b)

        # Apply transformation if function is provided
        if transform_vals is not None:
            value_data = [s.apply(transform_vals) for s in value_data]

        # Zip data
        value_data = zip(*value_data)

        # Make list of metadata series
        if use_meta:

            metadata_a = []

            if static_meta:
                for _ in range(len(values_a)):
                    metadata_a.append(
                        pd.Series(
                            [meta_a for _ in range(len(self.index))],
                            index=self.index[self._index_level_0]))
            elif generalize_meta_a:
                for _ in range(len(values_a)):
                    metadata_a.append(self._get_df_a_col(meta_a[0]))
            else:
                for name in meta_a:
                    metadata_a.append(self._get_df_a_col(name))

            metadata_b = []

            if static_meta:
                for _ in range(len(values_b)):
                    metadata_b.append(
                        pd.Series(
                            [meta_b for _ in range(len(self.index))],
                            index=self.index[self._index_level_1]))
            elif generalize_meta_b:
                for _ in range(len(values_b)):
                    metadata_b.append(self._get_df_b_col(meta_b[0]))
            else:
                for name in meta_b:
                    metadata_b.append(self._get_df_b_col(name))

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

        if use_meta:
            output = pd.Series(list(zip(value_data, metadata)))
        else:
            output = pd.Series(list(zip(value_data)))

        return output

    def keep_original(self,
                      columns_a,
                      columns_b,
                      suffix_a=None,
                      suffix_b=None,
                      sep='_'):
        """

        Specifies columns from df_a and df_b which should be included
        in the fused output, but which do not require conflict
        resolution.

        Parameters
        ----------
        columns_a : str/list
            A list of column names to be included from df_a.
        columns_b : str/list
            A list of column names to be included from df_b.
        suffix_a : str
            An optional suffix to be applied to the name of all
            columns kept from df_a.
        suffix_b : str
            An optional suffix to be applied to the names of all
            columns kept from df_b.
        sep : str
            The separator that should be used when resolving column
            name conflicts (e.g. with ``sep='_'``, ``taken_name``
            becomes ``taken_name_1``).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """

        columns_a = listify(columns_a)
        columns_b = listify(columns_b)

        for col in columns_a:
            if suffix_a is None:
                self.resolve(
                    None, [col], [],
                    name=col,
                    handler_override=KeepHandler(self))
            else:
                self.resolve(
                    None, [col], [],
                    name=col + sep + str(suffix_a),
                    handler_override=KeepHandler(self))

        for col in columns_b:
            if suffix_b is None:
                self.resolve(
                    None, [], [col],
                    name=col,
                    handler_override=KeepHandler(self))
            else:
                self.resolve(
                    None, [], [col],
                    name=col + sep + str(suffix_b),
                    handler_override=KeepHandler(self))

    def trust_your_friends(self,
                           values_a,
                           values_b,
                           trusted,
                           tie_break_trusted='random',
                           tie_break_untrusted='random',
                           label_a='a',
                           label_b='b',
                           name=None,
                           remove_na_vals=True):
        """
        Handles data conflicts by keeping data from a trusted source.

        Parameters
        ----------
        values_a : str/list
            Column name(s) from df_a to be resolved.
        values_b : str/list
            Column name(s) from df_b to be resolved.
        trusted : str
            The label of the preferred data source. By default, 'a'
            for df_a or 'b' for df_b.
        tie_break_trusted : str/function
            A conflict resolution function to be to break ties
            between trusted values.
        tie_break_untrusted : str/function
            A conflict resolution function to be to break ties
            between untrusted values.
        label_a : str/list
            The value(s) used to identify data from df_a. By default,
            all values are labelled 'a'. If multiple columns are
            specified in values_a, these may be identified with a
            list of column labels.
        label_b : str/list
            The value(s) used to identify data from df_b. By default,
            all values are labelled 'b'. If multiple columns are
            specified in values_b, these may be identified with a list
            of column labels.
        name : str
            The name of the resulting resolved column.
        remove_na_vals : bool
            If True, value/metadata pairs will be removed if the value
            is missing (i.e. np.nan).

        Returns
        -------
        None
            Queues a conflict resolution job, to be completed when
            ``fuse`` method is called.

        """
        self.resolve(
            choose_trusted,
            values_a,
            values_b,
            meta_a=label_a,
            meta_b=label_b,
            name=name,
            static_meta=True,
            params=trusted,
            remove_na_vals=remove_na_vals,
            remove_na_meta=False,
            tie_break=[
                process_tie_break(tie_break_trusted),
                process_tie_break(tie_break_untrusted)
            ],
            description='trust_your_friends')
