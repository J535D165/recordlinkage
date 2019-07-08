#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# testing utils from pandas
import pandas.util.testing as ptm

from recordlinkage import (OneToOneLinking, OneToManyLinking,
                           ConnectedComponents)


def test_one_to_one_linking_greedy():

    sample = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3), (3, 4), (3, 5),
                                        (4, 4), (5, 5), (6, 5), (7, 7), (7, 7),
                                        (7, 8)])
    one_to_many = OneToManyLinking()
    sample_one_to_many = one_to_many.compute(sample)

    expected = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3), (4, 4),
                                          (5, 5), (6, 5), (7, 7)])
    ptm.assert_index_equal(sample_one_to_many, expected)


def test_one_to_one_linking_max_weighted():

    sample_index = pd.MultiIndex.from_tuples([(1, 1), (2, 1), (2, 2), (2, 3), (3, 3)])
    sample_data = {"c1": [0, 1, 0, 1, 0], "c2": [1, 1, 1, 1, 1], "c3": [1, 1, 1, 1, 1]}
    sample = pd.DataFrame(data=sample_data, index=sample_index)

    one_to_one = OneToOneLinking(method="max_weighted")
    sample_one_to_one = one_to_one.compute(sample_index, sample)

    expected_index = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)])
    expected_data = {"weight": [2, 2, 2]}
    expected = pd.DataFrame(data=expected_data, index=expected_index)

    ptm.assert_frame_equal(sample_one_to_one, expected)


def test_one_to_many_linking():

    sample = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3), (3, 4), (3, 5),
                                        (4, 4), (5, 5), (6, 5), (7, 7), (7, 6),
                                        (7, 8)])

    # test OneToOneLinking
    one_to_one = OneToOneLinking()
    sample_one_to_one = one_to_one.compute(sample)

    expected = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3), (4, 4),
                                          (5, 5), (7, 7)])
    ptm.assert_index_equal(sample_one_to_one, expected)


def test_connected_components():

    sample = pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4), (5, 6), (5, 7),
                                        (8, 9)])

    # test ConnectedComponents
    connected = ConnectedComponents()
    sample_connected = connected.compute(sample)

    expected = [
        pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
        pd.MultiIndex.from_tuples([(5, 6), (5, 7)]),
        pd.MultiIndex.from_tuples([(8, 9)])
    ]

    for i, mi in enumerate(expected):
        ptm.assert_index_equal(sample_connected[i], expected[i])
