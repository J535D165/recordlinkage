from __future__ import division
from __future__ import unicode_literals

import multiprocessing as mp
import pandas as pd

from recordlinkage import logging
from recordlinkage.utils import listify
from recordlinkage.algorithms.conflict_resolution import (annotated_concat,
                                                          choose,
                                                          choose_max,
                                                          choose_min,
                                                          choose_metadata_max,
                                                          choose_metadata_min,
                                                          choose_random,
                                                          aggregate,
                                                          count,
                                                          group,
                                                          identity,
                                                          no_gossip,
                                                          vote)


class FuseCore(object):
    def __init__(self):
        self.vectors = None
        self.index = None
        self.predictions = None
        self.probabilities = None
        self.df_a = None
        self.df_b = None
        self.suffix_a = None
        self.suffix_b = None
        self.resolution_queue = []
        self._bases_taken = []
        self._names_taken = []
        self._sep = ''

    # Conflict Resolution Realizations

    def trust_your_friends(self, values_a, values_b, trusted, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str trusted: A preferred data source. 'a' for df_a or 'b' for df_b.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(choose, values_a, values_b, meta_a='a', meta_b='b', static_meta=True, params=(trusted,), name=name)

    def no_gossiping(self, values_a, values_b, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(no_gossip, values_a, values_b, name=name)

    def roll_the_dice(self, values_a, values_b, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(choose_random, values_a, values_b, name=name)

    def cry_with_the_wolves(self, values_a, values_b, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(vote, values_a, values_b, name=name)

    def pass_it_on(self, values_a, values_b, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(group, values_a, values_b, name=name)

    def meet_in_the_middle(self, values_a, values_b, metric, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param metric: The aggregation metric to be used. One of 'sum', 'mean', 'stdev', 'var'.
        :param name: The name of the resolved column.
        :return: None
        """
        # Metrics Available 2017-08-01: sum, mean, stdev, variance
        self.resolve(aggregate, values_a, values_b, params=(metric,), name=name)

    def keep_up_to_date(self, values_a, values_b, dates_a, dates_b, name=None):
        """

        :param str/list values_a: Column names from df_a to be resolved.
        :param str/list values_b: Column names from df_b to be resolved.
        :param str/list dates_a: Column names for dates in df_a.
        :param str/list dates_b: Column names for dates in df_b.
        :param name: The name of the resolved column.
        :return: None
        """
        self.resolve(choose_metadata_max, values_a, values_b, meta_a=dates_a, meta_b=dates_b, name=name)

    def resolve(self, fun, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                transform_meta=None, static_meta=False, params=None, name=None, **kwargs):
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
        :param function transform_vals: An optional pre-processing function to be applied to values.
        :param function transform_meta: An optional pre-processing function to be applied to metadata values.
        :param bool static_meta: If True, meta_a and meta_b values will be used as metadata values.
        :param params: Extra parameters used by the conflict resolution function.
        :param name: The name of the resolved column.
        :param kwargs: Optional keyword arguments.
        :return: A dictionary of metadata values.
        """
        # TODO: Add transformation function for value -> metadata mapping.
        # TODO: Optionally specify trusted columns.
        # NOTE: Optionally, could provide value columns as metadata columns (i.e. twice) and use transform_meta.
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        job = {
            'fun': fun,
            'values_a': values_a,
            'values_b': values_b,
            'meta_a': meta_a,
            'meta_b': meta_b,
            'transform_vals': transform_vals,
            'transform_meta': transform_meta,
            'static_meta': static_meta,
            'params': params,
            'name': name,
            'kwargs': kwargs
        }
        self.resolution_queue.append(job)
        return job

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None,
                                transform_vals=None, transform_meta=None, static_meta=False, **kwargs):
        """
        Formats data for conflict resolution. Output is a pandas.Series of nested tuples. _make_resolution_series
        is overriden by FuseLinks and FuseClusters. No implementation is provided in FuseCore.

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

    def _fusion_init(self, vectors, df_a, df_b, predictions, probabilities, sep):
        """
        A pre-fusion initialization routine to store the data inputs for access during
        the conflict resolution / data fusion process.

        :param pandas.DataFrame vectors: Multi-indexed comparison vectors i.e. produced by recordlinkage.Compare.
        :param pandas.DataFrame df_a: The original first data frame.
        :param pandas.DataFrame df_b: The original second data frame.
        :param pandas.Series predictions: A pandas.Series of True/False classifications.
        :param pandas.Series probabilities: A pandas.Series of candidate link probabilities (0 ≤ p ≤ 1).
        :param str sep: A string separator to be used in resolving column naming conflicts.
        :return: None
        """
        self.vectors = vectors
        self.index = vectors.index.to_frame()
        self.predictions = predictions
        self.probabilities = probabilities
        self.df_a = df_a
        self.df_b = df_b
        self._sep = sep

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

    def _do_resolve(self, job):
        """
        Perform conflict resolution for a queued job, by pre-processing data (_make_resolution_series) and performing
        the resolution (i.e. by applying a conflict resolution function).

        :param dict job: A dictionary of conflict resolution job metadata.
        :return: pandas.Series containing resolved/canonical values.
        """
        data = self._make_resolution_series(job['values_a'],
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

    def fuse(self, vectors, df_a, df_b, predictions=None, probabilities=None, n_cores=mp.cpu_count(), sep='_'):
        """
        Perform conflict resolution and data fusion for given data, using accumulated conflict resolution metadata.

        :param pandas.DataFrame vectors: Multi-indexed comparison vectors i.e. produced by recordlinkage.Compare.
        :param pandas.DataFrame df_a: The original first data frame.
        :param pandas.DataFrame df_b: The original second data frame.
        :param pandas.Series predictions: A pandas.Series of True/False classifications.
        :param pandas.Series probabilities: A pandas.Series of candidate link probabilities (0 ≤ p ≤ 1).
        :param n_cores: The number of cores to be used for processing. Defaults to all cores (mp.cpu_count()).
        :param str sep: A string separator to be used in resolving column naming conflicts.
        :return: A pandas.DataFrame with resolved/fused data.
        """

        if not isinstance(predictions, (type(None), pd.Series)):
            raise ValueError('Predictions must be a pandas Series.')

        if not isinstance(probabilities, (type(None), pd.Series)):
            raise ValueError('Predictions must be a pandas Series.')

        # Save references to input data.
        self._fusion_init(vectors, df_a, df_b, predictions, probabilities, sep)

        # Subclass-specific setup (e.g. applying refinements or detecting clusters).
        self._fusion_preprocess()

        # Resolve naming conflicts, if any.
        self._resolve_job_names(self._sep)

        # Compute resolved values for output.
        with mp.Pool(n_cores) as p:
            fused = p.map(self._do_resolve, self.resolution_queue)

        return pd.concat(fused, axis=1).set_index(self.index.index)


class FuseClusters(FuseCore):
    def __init__(self, method='???'):
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
    def __init__(self, unique_a=False, unique_b=False, rank_method=None, rank_links_by=None, rank_ascending=False):
        super().__init__()
        self.unique_a = unique_a
        self.unique_b = unique_b
        self.rank_method = rank_method
        if rank_links_by is not None:
            self.rank_links_by = listify(rank_links_by)
        else:
            self.rank_links_by = None
        self.rank_ascending = rank_ascending

    def _apply_predictions(self):

        # Turn predictions into desired formats
        pred_index = self.predictions.index
        pred_list = list(self.predictions)

        # Update vectors and indices
        self.vectors = self.vectors.iloc[pred_list]
        self.index = self.vectors.index.to_frame()

        # Refine data
        self.df_a = self.df_a.loc[pred_index.to_frame()[0]].iloc[pred_list].set_index(self.index.index)
        self.df_b = self.df_b.loc[pred_index.to_frame()[1]].iloc[pred_list].set_index(self.index.index)

        # Update predictions and probabilities
        if self.probabilities is not None:
            self.probabilities = self.probabilities.iloc[pred_list]
        if self.predictions is not None:
            self.predictions = self.predictions.iloc[pred_list]

    def _refine_predictions(self):
        pass

    def _fusion_preprocess(self):
        # if self.rank_method is not None:
        #     if self.predictions is None:
        #         self.predictions = pd.Series([True for _ in range(len(self.vectors))])
        #     print('start refine')
        #     self._refine_predictions()
        #     print('end refine')
        if self.predictions is not None:
            self._apply_predictions()

    def _get_df_a_col(self, name):
        return self.df_a[name].loc[list(self.index[0])]

    def _get_df_b_col(self, name):
        return self.df_b[name].loc[list(self.index[1])]

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                                transform_meta=None, static_meta=False, **kwargs):

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
                logging.warn('Generalizing values. There are fewer columns in values_a than in meta_a. '
                             'Values in first column of values_a will be generalized to values in meta_a.')
            elif len(values_a) > len(meta_a):
                generalize_values_a = False
                generalize_meta_a = True
                logging.warn('Generalizing metadata. There are fewer columns in meta_a than in values_a. '
                             'Values in first column of meta_a will be generalized to values in values_a.')
            else:
                generalize_values_a = False
                generalize_meta_a = False

            if len(values_b) < len(meta_b):
                generalize_values_b = True
                generalize_meta_b = False
                logging.warn('Generalizing values. There are fewer columns in values_b than in meta_b. '
                             'Values in first column of values_b will be generalized to values in meta_b.')
            elif len(values_b) > len(meta_b):
                generalize_values_b = False
                generalize_meta_b = True
                logging.warn('Generalizing metadata. There are fewer columns in meta_b than in values_b. '
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
                        pd.Series([meta_a for _ in range(len(self.index))], index=self.index[0])
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
                        pd.Series([meta_b for _ in range(len(self.index))], index=self.index[1])
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

    def keep(self, columns_a, columns_b, suffix_a=None, suffix_b=None, sep='_'):
        # Add "keeps" to a new queue of columns — analogous but distinct from the resolution queue

        columns_a = listify(columns_a)
        columns_b = listify(columns_b)

        for col in columns_a:
            if suffix_a is None:
                self.resolve(identity, col, [], name=col)
            else:
                self.resolve(identity, col, [], name=col + sep + str(suffix_a))

        for col in columns_b:
            if suffix_b is None:
                self.resolve(identity, [], col, name=col)
            else:
                self.resolve(identity, [], col, name=col + sep + str(suffix_b))
