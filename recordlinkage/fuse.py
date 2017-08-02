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
                                                          compute_metric,
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
        self.df_a = None
        self.df_b = None
        self.suffix_a = None
        self.suffix_b = None
        self.resolution_queue = []
        self.bases_taken = []
        self.names_taken = []
        self.sep = ''

    # Conflict Resolution Realizations

    def trust_your_friends(self, c1, c2, trusted, name=None):
        self._queue_resolve(choose, c1, c2, meta_a='a', meta_b='b', static_meta=True, params=(trusted,), name=name)

    def no_gossiping(self, c1, c2, name=None):
        self._queue_resolve(no_gossip, c1, c2, name=name)

    def roll_the_dice(self, c1, c2, name=None):
        self._queue_resolve(choose_random, c1, c2, name=name)

    def cry_with_the_wolves(self, c1, c2, name=None):
        self._queue_resolve(vote, c1, c2, name=name)

    def pass_it_on(self, c1, c2, name=None):
        self._queue_resolve(group, c1, c2, name=name)

    def meet_in_the_middle(self, c1, c2, metric, name=None):
        # Metrics Available 2017-08-01: sum, mean, stdev, variance
        self._queue_resolve(compute_metric, c1, c2, params=(metric,), name=name)

    def keep_up_to_date(self, c1, c2, dates_a, dates_b, name=None):
        self._queue_resolve(choose_metadata_max, c1, c2, meta_a=dates_a, meta_b=dates_b, name=name)

    def resolve(self, fun, data, params):
        """
        Perform conflict resolution on reorganized data.

        :param function fun: A conflict resolution function.
        :param pd.Series data: A pandas Series of nested tuples.
        :param tuple params: A list of extra parameters to be passed to the conflict resolution function.
        :return: pd.Series of resolved values.
        """
        return data.apply(fun, args=params)

    def _queue_resolve(self, fun, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                       transform_meta=None, static_meta=False, params=None, name=None, **kwargs):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        # TODO: Add column names for output
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
        # No implementation provided.
        # Override in subclass.
        return pd.Series()

    def _fusion_init(self, vectors, df_a, df_b, predictions, sep):
        self.vectors = vectors
        self.index = vectors.index.to_frame()
        self.predictions = predictions
        self.df_a = df_a
        self.df_b = df_b
        self.sep = sep

    def _fusion_setup(self):
        # No implementation provided.
        # Override in subclass.
        pass

    def _resolve_job_names(self, sep):
        for job in self.resolution_queue:
            if job['name'] is not None:
                if job['name'] not in self.bases_taken:
                    if job['name'] not in self.names_taken:
                        self.bases_taken.append(job['name'])
                        self.names_taken.append(job['name'])
                else:
                    i = 1
                    while True:
                        name = str(job['name']) + sep + str(i)
                        if name in self.names_taken:
                            if i > 1000:
                                raise RuntimeError('Data fusion hung while attempting to resolve column names.'
                                                   '1000 name suffixes were attempted.'
                                                   'Check for excessive name conflicts.')
                            else:
                                i += 1
                        else:
                            self.names_taken.append(name)
                            job['name'] = name
                            break

    def _do_resolve(self, job):
        data = self.resolve(job['fun'],
                            self._make_resolution_series(job['values_a'],
                                                         job['values_b'],
                                                         meta_a=job['meta_a'],
                                                         meta_b=job['meta_b'],
                                                         transform_vals=job['transform_vals'],
                                                         transform_meta=job['transform_meta'],
                                                         static_meta=job['static_meta']),
                            job['params']).rename(job['name'])
        if job['name'] is not None:
            data.rename(job['name'])
        return data

    def fuse(self, vectors, df_a, df_b, predictions=None, n_cores=mp.cpu_count(), sep='_'):

        # Apply refinements to vectors / index
        # Make calls to `resolve` using accumulated metadata
        # Return the fused data frame

        # Save references to input data.
        self._fusion_init(vectors, df_a, df_b, predictions, sep)

        # Subclass-specific setup (e.g. applying refinements or detecting clusters).
        self._fusion_setup()

        # Resolve naming conflicts, if any.
        self._resolve_job_names(self.sep)

        # Compute resolved values for output.
        with mp.Pool(n_cores) as p:
            fused = p.map(self._do_resolve, self.resolution_queue)

        return pd.concat(fused, axis=1)


class FuseClusters(FuseCore):
    def __init__(self, method='???'):
        super().__init__()
        self.method = method

    def _find_clusters(self, method):
        pass

    def _fusion_setup(self):
        pass

    def _make_resolution_series(self, values_a, values_b, meta_a=None, meta_b=None, transform_vals=None,
                                transform_meta=None, static_meta=False, **kwargs):
        pass


class FuseLinks(FuseCore):
    def __init__(self, unique_a=False, unique_b=False):
        super().__init__()
        self.unique_a = unique_a
        self.unique_b = unique_b

    def _apply_refinement(self):
        pass

    def _fusion_setup(self):
        pass

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
                self._queue_resolve(identity, col, [], name=col)
            else:
                self._queue_resolve(identity, col, [], name=col+sep+str(suffix_a))

        for col in columns_b:
            if suffix_b is None:
                self._queue_resolve(identity, [], col, name=col)
            else:
                self._queue_resolve(identity, [], col, name=col+sep+str(suffix_b))
