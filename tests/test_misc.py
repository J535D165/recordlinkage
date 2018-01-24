import pandas as pd
import numpy as np

# testing utils from pandas
import pandas.util.testing as ptm

from recordlinkage import index_split


def test_multiindex_split():

    index = pd.MultiIndex.from_product([np.arange(5), np.arange(6)])
    result = index_split(index, 3)

    assert len(result) == 3

    for i, result_index_chunk in enumerate(result):
        expected_index_chunk = index[i * 10:(i + 1) * 10]
        ptm.assert_index_equal(result_index_chunk, expected_index_chunk)

        assert len(result_index_chunk.levels) == 2
        assert len(result_index_chunk.labels) == 2
