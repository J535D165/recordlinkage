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
        self.resolution_queue = []

    def resolve(self, fun, data):
        # Integrate values in vals (Series of tuples) using optional
        # meta (Series of tuples) by applying the conflict resolution function fun
        # to the series of tuples.
        pass

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

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, **kwargs):
        # No implementation provided.
        # Override in subclass.
        pass

    # Conflict Resolution Realizations

    def trust_your_friends(self, c1, c2, trusted):
        self.queue_resolve(choose, c1, c2, trust=trusted)

    def no_gossiping(self, c1, c2):
        self.queue_resolve(no_gossip, c1, c2)

    def roll_the_dice(self, c1, c2):
        self.queue_resolve(choose_random, c1, c2)

    # Turn collected metadata into a final result.
    def fuse(self, vectors, df_a, df_b, suffix_a='_a', suffix_b='_b'):
        # Apply refinements to vectors / index
        # Make calls to `resolve` using accumulated metadata
        # Return the fused data frame
        pass


class FuseClusters(FuseCore):
    def __init__(self, method='???'):
        super().__init__()
        self.method = method

    def _find_clusters(self, method):
        pass

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass


class FuseLinks(FuseCore):
    def __init__(self, unique_a=False, unique_b=False):
        super().__init__()
        self.unique_a = unique_a
        self.unique_b = unique_b

    def _apply_refinement(self):
        pass

    def _format_resolve(self, c1, c2, m1=None, m2=None, trans_c=None, trans_m=None, *args, **kwargs):
        pass
