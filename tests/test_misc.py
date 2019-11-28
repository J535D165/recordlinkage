#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# testing utils from pandas
import pandas.util.testing as ptm
import pytest

import recordlinkage as rl
from recordlinkage import index_split
from recordlinkage.utils import is_min_pandas_version


def test_multiindex_split():

    index = pd.MultiIndex.from_product([np.arange(5), np.arange(6)])
    result = index_split(index, 3)

    assert len(result) == 3

    for i, result_index_chunk in enumerate(result):
        expected_index_chunk = index[i * 10:(i + 1) * 10]
        ptm.assert_index_equal(result_index_chunk, expected_index_chunk)

        assert len(result_index_chunk.levels) == 2
        if is_min_pandas_version("0.24.0"):
            assert len(result_index_chunk.codes) == 2
        else:
            assert len(result_index_chunk.labels) == 2


def test_options():

    # global set
    rl.options.indexing.pairs = "multiindex"
    assert rl.get_option("indexing.pairs") == "multiindex"


def test_options_context():

    with rl.option_context("indexing.pairs", "multiindex"):
        rl.options.indexing.pairs = "multiindex"
        assert rl.get_option("indexing.pairs") == "multiindex"


def test_options_incorrect_values():
    # incorrect value
    with pytest.raises(ValueError):
        rl.options.indexing.pairs = "non_existing"
