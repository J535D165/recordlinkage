from __future__ import division
from __future__ import unicode_literals

import random
import warnings
import collections
import statistics
import multiprocessing as mp
import pandas as pd
import numpy as np

from recordlinkage.algorithms.conflict_resolution import (annotated_concat,
                                                          choose,
                                                          choose_max,
                                                          choose_min,
                                                          choose_random,
                                                          compute_metric,
                                                          count,
                                                          group,
                                                          maximize_metadata_value,
                                                          minimize_metadata_value,
                                                          no_gossip,
                                                          vote)


class FuseCore(object):
    def __init__(self):
        self.vectors = None
        self.df_a = None
        self.df_b = None
        self.suffix_a = None
        self.suffix_b = None
        self.resolution_queue = []

    def resolve(self, fun, data):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        return data.apply(fun)

    def queue_resolve(self, fun, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, **kwargs):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        self.resolution_queue.append(
            {
                'fun': fun,
                'c1': c1,
                'c2': c2,
                'm1': m1,
                'm2': m2,
                'trans_c': trans_c,
                'trans_m': trans_m,
                'kwargs': kwargs
            }
        )

    def _prep_resolution_data(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, **kwargs):
        # No implementation provided.
        # Override in subclass.
        return pd.Series()

    # Conflict Resolution Realizations

    def trust_your_friends(self, c1, c2, trusted):
        self.queue_resolve(choose, c1, c2, trust=trusted)

    def no_gossiping(self, c1, c2):
        self.queue_resolve(no_gossip, c1, c2)

    def roll_the_dice(self, c1, c2):
        self.queue_resolve(choose_random, c1, c2)

    def prep_fuse(self):
        # No implementation provided.
        # Override in subclass.
        pass

    # Turn collected metadata into a final result.
    def fuse(self, vectors, df_a, df_b, suffix_a='_a', suffix_b='_b'):
        # Apply refinements to vectors / index
        # Make calls to `resolve` using accumulated metadata
        # Return the fused data frame

        # Save references to input data.
        self.vectors = vectors
        self.df_a = df_a
        self.df_b = df_b
        self.suffix_a = suffix_a
        self.suffix_b = suffix_b

        # Subclass-specific data fusion preparation.
        self.prep_fuse()

        # Compute resolved values for output.
        # TODO: Optionally include comparison vectors.
        # TODO: Optionally include pre-resolution column data.
        # TODO: Optionally include non-resolved column data.

        fused = []

        for job in self.resolution_queue:
            fused.append(
                self.resolve(job['fun'],
                             self._prep_resolution_data(job['c1'],
                                                        job['c2'],
                                                        m1=job['m1'],
                                                        m2=job['m2'],
                                                        trans_c=job['trans_c'],
                                                        trans_m=job['trans_m'],
                                                        **job['kwargs']))
            )

        return pd.concat(fused)


class FuseClusters(FuseCore):
    def __init__(self, method='???'):
        super().__init__()
        self.method = method

    def _find_clusters(self, method):
        pass

    def prep_fuse(self):
        pass

    def _prep_resolution_data(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass


class FuseLinks(FuseCore):
    def __init__(self, unique_a=False, unique_b=False):
        super().__init__()
        self.unique_a = unique_a
        self.unique_b = unique_b

    def _apply_refinement(self):
        pass

    def prep_fuse(self):
        pass

    def _prep_resolution_data(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass
