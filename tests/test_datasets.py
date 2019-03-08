#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

import pandas

import pytest

from recordlinkage.datasets import (load_febrl1, load_febrl2, load_febrl3,
                                    load_febrl4, load_krebsregister,
                                    binary_vectors)


FEBRL_DEDUP = [
    # nlinks = 500
    (load_febrl1, 1000, 500),
    # nlinks=19*6*5/2+47*5*4/2+107*4*3/2+141*3*2/2+114
    (load_febrl2, 5000, 1934),
    # nlinks=168*6*5/2+161*5*4/2+212*4*3/2+256*3*2/2+368
    (load_febrl3, 5000, 6538)
]


class TestExternalDatasets(object):

    @pytest.mark.parametrize("dataset,nrows,nlinks", FEBRL_DEDUP)
    def test_febrl_dedup(self, dataset, nrows, nlinks):

        df = dataset()
        assert isinstance(df, pandas.DataFrame)
        assert len(df) == nrows

    @pytest.mark.parametrize("dataset,nrows,nlinks", FEBRL_DEDUP)
    def test_febrl_dedup_links(self, dataset, nrows, nlinks):

        df, links = dataset(return_links=True)
        assert isinstance(df, pandas.DataFrame)
        assert len(df) == nrows
        assert len(links) == nlinks
        assert isinstance(links, pandas.MultiIndex)

    @pytest.mark.parametrize("dataset,nrows,nlinks", FEBRL_DEDUP)
    def test_febrl_dedup_tril(self, dataset, nrows, nlinks):

        df, links = dataset(return_links=True)

        s_level_1 = pandas.Series(numpy.arange(len(df)), index=df.index)
        s_level_2 = pandas.Series(numpy.arange(len(df)), index=df.index)

        x1 = s_level_1.loc[links.get_level_values(0)]
        x2 = s_level_2.loc[links.get_level_values(1)]

        assert numpy.all(x1.values > x2.values)

    def test_febrl4(self):

        dfa, dfb = load_febrl4()
        assert isinstance(dfa, pandas.DataFrame)
        assert isinstance(dfb, pandas.DataFrame)
        assert len(dfa) == 5000
        assert len(dfb) == 5000

    def test_febrl_links(self):
        dfa, dfb, links = load_febrl4(return_links=True)
        assert isinstance(dfa, pandas.DataFrame)
        assert isinstance(dfb, pandas.DataFrame)
        assert len(dfa) == 5000
        assert len(dfb) == 5000
        assert isinstance(links, pandas.MultiIndex)

    def test_krebs_dataset(self):

        krebs_data, krebs_matches = load_krebsregister()
        krebs_data_block1, krebs_matches_block1 = load_krebsregister(1)
        krebs_data_block10, krebs_matches_block10 = load_krebsregister(10)

        # count the number of recordss
        assert type(krebs_data), pandas.DataFrame
        assert type(krebs_matches), pandas.MultiIndex
        assert len(krebs_data) == 5749132
        assert len(krebs_matches) == 20931

        assert len(krebs_data_block1) > 0
        assert len(krebs_data_block10) > 0

        # load not existing block
        with pytest.raises(ValueError):
            load_krebsregister(11)

        # missing values
        krebs_block10, matches = load_krebsregister(10, missing_values=0)
        assert krebs_block10.isnull().sum().sum() == 0

    def test_krebs_missings(self):

        # missing values
        krebs_block10, matches = load_krebsregister(10, missing_values=0)
        assert krebs_block10.isnull().sum().sum() == 0

    def test_krebs_shuffle(self):

        # missing values
        krebs_block10, matches = load_krebsregister(10, shuffle=False)


class TestGeneratedDatasets(object):
    def test_random_comparison_vectors(self):
        # Test the generation of a random dataset

        n_record_pairs = 10000
        n_matches = 500

        df = binary_vectors(
            n_record_pairs,
            n_matches,
            m=[0.8] * 8,
            u=[0.2] * 8,
            random_state=535)

        # Check the result is a DataFrame with MultiIndex
        assert isinstance(df, pandas.DataFrame)
        assert isinstance(df.index, pandas.MultiIndex)

        # Test the length of the dataframe
        assert len(df) == n_record_pairs
