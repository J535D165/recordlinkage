#!/usr/bin/env python

import sys

import pandas as pd

# testing utils from pandas
import pandas.testing as pdt
import pytest

try:
    import networkx  # noqa
except ImportError:
    pass

from recordlinkage import ConnectedComponents
from recordlinkage import OneToManyLinking
from recordlinkage import OneToOneLinking


def test_one_to_one_linking():
    sample = pd.MultiIndex.from_tuples(
        [
            (1, 1),
            (2, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 4),
            (5, 5),
            (6, 5),
            (7, 7),
            (7, 7),
            (7, 8),
        ]
    )
    one_to_many = OneToManyLinking()
    sample_one_to_many = one_to_many.compute(sample)

    expected = pd.MultiIndex.from_tuples(
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 5), (7, 7)]
    )
    pdt.assert_index_equal(sample_one_to_many, expected)


def test_one_to_many_linking():
    sample = pd.MultiIndex.from_tuples(
        [
            (1, 1),
            (2, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 4),
            (5, 5),
            (6, 5),
            (7, 7),
            (7, 6),
            (7, 8),
        ]
    )

    # test OneToOneLinking
    one_to_one = OneToOneLinking()
    sample_one_to_one = one_to_one.compute(sample)

    expected = pd.MultiIndex.from_tuples(
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]
    )
    pdt.assert_index_equal(sample_one_to_one, expected)


@pytest.mark.skipif(
    "networkx" not in sys.modules, reason="Requires the Networkx library"
)
def test_connected_components():
    sample = pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4), (5, 6), (5, 7), (8, 9)])

    # test ConnectedComponents
    connected = ConnectedComponents()
    sample_connected = connected.compute(sample)

    expected = [
        pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
        pd.MultiIndex.from_tuples([(5, 6), (5, 7)]),
        pd.MultiIndex.from_tuples([(8, 9)]),
    ]

    for i, _mi in enumerate(expected):
        pdt.assert_index_equal(sample_connected[i], expected[i])
