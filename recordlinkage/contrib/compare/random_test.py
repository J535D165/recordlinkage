#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import recordlinkage
from recordlinkage.index import Full
from recordlinkage.contrib.compare import RandomContinuous


class TestRandomContinuous(object):

    def test_random_cont_standalone(self):

        arr1 = [1, 2, 3, 4, 5]
        arr2 = [1, 2, 3, 4, 5]
        pairs = pd.MultiIndex.from_product([arr1, arr2])

        c = RandomContinuous()
        r = c.compute(pairs)

        assert r.shape[0] == len(arr1) * len(arr2)

    def test_random_cont(self):

        df_a = pd.DataFrame({'v': list("abcde")})
        df_b = pd.DataFrame({'v': list("abcde")})

        pairs = Full().index(df_a, df_b)

        c = recordlinkage.Compare()
        c.exact("v", "v")
        c.add(RandomContinuous(label='random'))
        cv = c.compute(pairs, df_a, df_b)

        assert isinstance(cv, pd.DataFrame)

        assert cv['random'].notnull().all()
        assert cv['random'].min() >= 0.0
        assert cv['random'].max() <= 1.0
