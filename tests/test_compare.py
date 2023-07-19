#!/usr/bin/env python

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas.testing as pdt

# dependencies testing specific
import pytest
from numpy import arange
from numpy import nan
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series
from pandas import to_datetime

import recordlinkage
from recordlinkage.base import BaseCompareFeature

STRING_SIM_ALGORITHMS = [
    "jaro",
    "q_gram",
    "cosine",
    "jaro_winkler",
    "dameraulevenshtein",
    "levenshtein",
    "lcs",
    "smith_waterman",
]

NUMERIC_SIM_ALGORITHMS = ["step", "linear", "squared", "exp", "gauss"]

FIRST_NAMES = [
    "Ronald",
    "Amy",
    "Andrew",
    "William",
    "Frank",
    "Jessica",
    "Kevin",
    "Tyler",
    "Yvonne",
    nan,
]
LAST_NAMES = [
    "Graham",
    "Smith",
    "Holt",
    "Pope",
    "Hernandez",
    "Gutierrez",
    "Rivera",
    nan,
    "Crane",
    "Padilla",
]
STREET = [
    "Oliver Neck",
    nan,
    "Melissa Way",
    "Sara Dale",
    "Keith Green",
    "Olivia Terrace",
    "Williams Trail",
    "Durham Mountains",
    "Anna Circle",
    "Michelle Squares",
]
JOB = [
    "Designer, multimedia",
    "Designer, blown glass/stained glass",
    "Chiropractor",
    "Engineer, mining",
    "Quantity surveyor",
    "Phytotherapist",
    "Teacher, English as a foreign language",
    "Electrical engineer",
    "Research officer, government",
    "Economist",
]
AGES = [23, 40, 70, 45, 23, 57, 38, nan, 45, 46]

# Run all tests in this file with:
# nosetests tests/test_compare.py


class TestData:
    @classmethod
    def setup_class(cls):
        N_A = 100
        N_B = 100

        cls.A = DataFrame(
            {
                "age": np.random.choice(AGES, N_A),
                "given_name": np.random.choice(FIRST_NAMES, N_A),
                "lastname": np.random.choice(LAST_NAMES, N_A),
                "street": np.random.choice(STREET, N_A),
            }
        )

        cls.B = DataFrame(
            {
                "age": np.random.choice(AGES, N_B),
                "given_name": np.random.choice(FIRST_NAMES, N_B),
                "lastname": np.random.choice(LAST_NAMES, N_B),
                "street": np.random.choice(STREET, N_B),
            }
        )

        cls.A.index.name = "index_df1"
        cls.B.index.name = "index_df2"

        cls.index_AB = MultiIndex.from_arrays(
            [arange(len(cls.A)), arange(len(cls.B))],
            names=[cls.A.index.name, cls.B.index.name],
        )

        # Create a temporary directory
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        # Remove the test directory
        shutil.rmtree(cls.test_dir)


class TestCompareApi(TestData):
    """General unittest for the compare API."""

    def test_repr(self):
        comp = recordlinkage.Compare()
        comp.exact("given_name", "given_name")
        comp.string("given_name", "given_name", method="jaro")
        comp.numeric("age", "age", method="step", offset=3, origin=2)
        comp.numeric("age", "age", method="step", offset=0, origin=2)

        c_str = str(comp)
        c_repr = repr(comp)
        assert c_str == c_repr

        start_str = f"<{comp.__class__.__name__}"
        assert c_str.startswith(start_str)

    def test_instance_linking(self):
        comp = recordlinkage.Compare()
        comp.exact("given_name", "given_name")
        comp.string("given_name", "given_name", method="jaro")
        comp.numeric("age", "age", method="step", offset=3, origin=2)
        comp.numeric("age", "age", method="step", offset=0, origin=2)
        result = comp.compute(self.index_AB, self.A, self.B)

        # returns a Series
        assert isinstance(result, DataFrame)

        # resulting series has a MultiIndex
        assert isinstance(result.index, MultiIndex)

        # indexnames are oke
        assert result.index.names == [self.A.index.name, self.B.index.name]

        assert len(result) == len(self.index_AB)

    def test_instance_dedup(self):
        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.numeric("age", "age", method="step", offset=3, origin=2)
        comp.numeric("age", "age", method="step", offset=0, origin=2)
        result = comp.compute(self.index_AB, self.A)

        # returns a Series
        assert isinstance(result, DataFrame)

        # resulting series has a MultiIndex
        assert isinstance(result.index, MultiIndex)

        # indexnames are oke
        assert result.index.names == [self.A.index.name, self.B.index.name]

        assert len(result) == len(self.index_AB)

    def test_label_linking(self):
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int),
            "given_name",
            "given_name",
            label="my_feature_label",
        )
        result = comp.compute(self.index_AB, self.A, self.B)

        assert "my_feature_label" in result.columns.tolist()

    def test_label_dedup(self):
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int),
            "given_name",
            "given_name",
            label="my_feature_label",
        )
        result = comp.compute(self.index_AB, self.A)

        assert "my_feature_label" in result.columns.tolist()

    def test_multilabel_none_linking(self):
        def ones_np_multi(s1, s2):
            return np.ones(len(s1)), np.ones((len(s1), 3))

        def ones_pd_multi(s1, s2):
            return (Series(np.ones(len(s1))), DataFrame(np.ones((len(s1), 3))))

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(ones_np_multi, "given_name", "given_name")
        comp.compare_vectorized(ones_pd_multi, "given_name", "given_name")
        result = comp.compute(self.index_AB, self.A, self.B)

        assert [0, 1, 2, 3, 4, 5, 6, 7, 8] == result.columns.tolist()

    def test_multilabel_linking(self):
        def ones_np_multi(s1, s2):
            return np.ones(len(s1)), np.ones((len(s1), 3))

        def ones_pd_multi(s1, s2):
            return (Series(np.ones(len(s1))), DataFrame(np.ones((len(s1), 3))))

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(
            ones_np_multi, "given_name", "given_name", label=["a", ["b", "c", "d"]]
        )
        comp.compare_vectorized(
            ones_pd_multi, "given_name", "given_name", label=["e", ["f", "g", "h"]]
        )
        result = comp.compute(self.index_AB, self.A, self.B)

        assert [0, "a", "b", "c", "d", "e", "f", "g", "h"] == result.columns.tolist()

    def test_multilabel_dedup(self):
        def ones_np_multi(s1, s2):
            return np.ones(len(s1)), np.ones((len(s1), 3))

        def ones_pd_multi(s1, s2):
            return (Series(np.ones(len(s1))), DataFrame(np.ones((len(s1), 3))))

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(
            ones_np_multi, "given_name", "given_name", label=["a", ["b", "c", "d"]]
        )
        comp.compare_vectorized(
            ones_pd_multi, "given_name", "given_name", label=["e", ["f", "g", "h"]]
        )
        result = comp.compute(self.index_AB, self.A)

        assert [0, "a", "b", "c", "d", "e", "f", "g", "h"] == result.columns.tolist()

    def test_multilabel_none_dedup(self):
        def ones_np_multi(s1, s2):
            return np.ones(len(s1)), np.ones((len(s1), 3))

        def ones_pd_multi(s1, s2):
            return (Series(np.ones(len(s1))), DataFrame(np.ones((len(s1), 3))))

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(ones_np_multi, "given_name", "given_name")
        comp.compare_vectorized(ones_pd_multi, "given_name", "given_name")
        result = comp.compute(self.index_AB, self.A)

        assert [0, 1, 2, 3, 4, 5, 6, 7, 8] == result.columns.tolist()

    def test_multilabel_error_dedup(self):
        def ones(s1, s2):
            return np.ones((len(s1), 2))

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(ones, "given_name", "given_name", label=["a", "b", "c"])

        with pytest.raises(ValueError):
            comp.compute(self.index_AB, self.A)

    def test_incorrect_collabels_linking(self):
        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name", method="jaro")
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int),
            "given_name",
            "not_existing_label",
        )

        with pytest.raises(KeyError):
            comp.compute(self.index_AB, self.A, self.B)

    def test_incorrect_collabels_dedup(self):
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int),
            "given_name",
            "not_existing_label",
        )

        with pytest.raises(KeyError):
            comp.compute(self.index_AB, self.A)

    def test_compare_custom_vectorized_linking(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int), "col", "col"
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int),
            "col",
            "col",
            label="my_feature_label",
        )

        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix, columns=["my_feature_label"])
        pdt.assert_frame_equal(result, expected)

    # def test_compare_custom_nonvectorized_linking(self):

    #     A = DataFrame({'col': [1, 2, 3, 4, 5]})
    #     B = DataFrame({'col': [1, 2, 3, 4, 5]})
    #     ix = MultiIndex.from_arrays([A.index.values, B.index.values])

    #     def custom_func(a, b):
    #         return np.int64(1)

    #     # test without label
    #     comp = recordlinkage.Compare()
    #     comp.compare_single(
    #         custom_func,
    #         'col',
    #         'col'
    #     )
    #     result = comp.compute(ix, A, B)
    #     expected = DataFrame([1, 1, 1, 1, 1], index=ix)
    #     pdt.assert_frame_equal(result, expected)

    #     # test with label
    #     comp = recordlinkage.Compare()
    #     comp.compare_single(
    #         custom_func,
    #         'col',
    #         'col',
    #         label='test'
    #     )

    #     result = comp.compute(ix, A, B)
    #     expected = DataFrame([1, 1, 1, 1, 1], index=ix, columns=['test'])
    #     pdt.assert_frame_equal(result, expected)

    def test_compare_custom_instance_type(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        def call(s1, s2):
            # this should raise on incorrect types
            assert isinstance(s1, np.ndarray)
            assert isinstance(s2, np.ndarray)

            return np.ones(len(s1), dtype=int)

        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int), "col", "col"
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_arguments_linking(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x: np.ones(len(s1), dtype=int) * x, "col", "col", 5
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x: np.ones(len(s1), dtype=int) * x,
            "col",
            "col",
            5,
            label="test",
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=["test"])
        pdt.assert_frame_equal(result, expected)

        # test with kwarg
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x: np.ones(len(s1), dtype=int) * x,
            "col",
            "col",
            x=5,
            label="test",
        )
        result = comp.compute(ix, A, B)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=["test"])
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_dedup(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        ix = MultiIndex.from_arrays([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int), "col", "col"
        )
        result = comp.compute(ix, A)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2: np.ones(len(s1), dtype=int), "col", "col", label="test"
        )
        result = comp.compute(ix, A)
        expected = DataFrame([1, 1, 1, 1, 1], index=ix, columns=["test"])
        pdt.assert_frame_equal(result, expected)

    def test_compare_custom_vectorized_arguments_dedup(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        ix = MultiIndex.from_arrays([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])

        # test without label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x: np.ones(len(s1), dtype=int) * x, "col", "col", 5
        )
        result = comp.compute(ix, A)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix)
        pdt.assert_frame_equal(result, expected)

        # test with label
        comp = recordlinkage.Compare()
        comp.compare_vectorized(
            lambda s1, s2, x: np.ones(len(s1), dtype=int) * x,
            "col",
            "col",
            5,
            label="test",
        )
        result = comp.compute(ix, A)
        expected = DataFrame([5, 5, 5, 5, 5], index=ix, columns=["test"])
        pdt.assert_frame_equal(result, expected)

    def test_parallel_comparing_api(self):
        # use single job
        comp = recordlinkage.Compare(n_jobs=1)
        comp.exact("given_name", "given_name", label="my_feature_label")
        result_single = comp.compute(self.index_AB, self.A, self.B)
        result_single.sort_index(inplace=True)

        # use two jobs
        comp = recordlinkage.Compare(n_jobs=2)
        comp.exact("given_name", "given_name", label="my_feature_label")
        result_2processes = comp.compute(self.index_AB, self.A, self.B)
        result_2processes.sort_index(inplace=True)

        # compare results
        pdt.assert_frame_equal(result_single, result_2processes)

    def test_parallel_comparing(self):
        # use single job
        comp = recordlinkage.Compare(n_jobs=1)
        comp.exact("given_name", "given_name", label="my_feature_label")
        result_single = comp.compute(self.index_AB, self.A, self.B)
        result_single.sort_index(inplace=True)

        # use two jobs
        comp = recordlinkage.Compare(n_jobs=2)
        comp.exact("given_name", "given_name", label="my_feature_label")
        result_2processes = comp.compute(self.index_AB, self.A, self.B)
        result_2processes.sort_index(inplace=True)

        # use two jobs
        comp = recordlinkage.Compare(n_jobs=4)
        comp.exact("given_name", "given_name", label="my_feature_label")
        result_4processes = comp.compute(self.index_AB, self.A, self.B)
        result_4processes.sort_index(inplace=True)

        # compare results
        pdt.assert_frame_equal(result_single, result_2processes)
        pdt.assert_frame_equal(result_single, result_4processes)

    def test_pickle(self):
        # test if it is possible to pickle the Compare class

        comp = recordlinkage.Compare()
        comp.string("given_name", "given_name")
        comp.numeric("number", "number")
        comp.geo("lat", "lng", "lat", "lng")
        comp.date("before", "after")

        # do the test
        pickle_path = os.path.join(self.test_dir, "pickle_compare_obj.pickle")
        pickle.dump(comp, open(pickle_path, "wb"))

    def test_manual_parallel_joblib(self):
        # test if it is possible to pickle the Compare class
        # This is only available for python 3. For python 2, it is not
        # possible to pickle instancemethods. A workaround can be found at
        # https://stackoverflow.com/a/29873604/8727928

        if sys.version.startswith("3"):
            # import joblib dependencies
            from joblib import Parallel
            from joblib import delayed

            # split the data into smaller parts
            len_index = int(len(self.index_AB) / 2)
            df_chunks = [self.index_AB[0:len_index], self.index_AB[len_index:]]

            comp = recordlinkage.Compare()
            comp.string("given_name", "given_name")
            comp.string("lastname", "lastname")
            comp.exact("street", "street")

            # do in parallel
            Parallel(n_jobs=2)(
                delayed(comp.compute)(df_chunks[i], self.A, self.B) for i in [0, 1]
            )

    def test_indexing_types(self):
        # test the two types of indexing

        # this test needs improvement

        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B_reversed = B[::-1].copy()
        ix = MultiIndex.from_arrays([np.arange(5), np.arange(5)])

        # test with label indexing type
        comp_label = recordlinkage.Compare(indexing_type="label")
        comp_label.exact("col", "col")
        result_label = comp_label.compute(ix, A, B_reversed)

        # test with position indexing type
        comp_position = recordlinkage.Compare(indexing_type="position")
        comp_position.exact("col", "col")
        result_position = comp_position.compute(ix, A, B_reversed)

        assert (result_position.values == 1).all(axis=0)

        pdt.assert_frame_equal(result_label, result_position)

    def test_pass_list_of_features(self):
        from recordlinkage.compare import FrequencyA
        from recordlinkage.compare import VariableA
        from recordlinkage.compare import VariableB

        # setup datasets and record pairs
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        ix = MultiIndex.from_arrays([np.arange(5), np.arange(5)])

        # test with label indexing type

        features = [
            VariableA("col", label="y1"),
            VariableB("col", label="y2"),
            FrequencyA("col", label="y3"),
        ]
        comp_label = recordlinkage.Compare(features=features)
        result_label = comp_label.compute(ix, A, B)

        assert list(result_label) == ["y1", "y2", "y3"]


class TestCompareFeatures(TestData):
    def test_feature(self):
        # test using classes and the base class

        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        feature = BaseCompareFeature("col", "col")
        feature._f_compare_vectorized = lambda s1, s2: np.ones(len(s1))
        feature.compute(ix, A, B)

    def test_feature_multicolumn_return(self):
        # test using classes and the base class

        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        def ones(s1, s2):
            return DataFrame(np.ones((len(s1), 3)))

        feature = BaseCompareFeature("col", "col")
        feature._f_compare_vectorized = ones
        result = feature.compute(ix, A, B)

        assert result.shape == (5, 3)

    def test_feature_multicolumn_input(self):
        # test using classes and the base class

        A = DataFrame(
            {
                "col1": ["abc", "abc", "abc", "abc", "abc"],
                "col2": ["abc", "abc", "abc", "abc", "abc"],
            }
        )
        B = DataFrame(
            {
                "col1": ["abc", "abd", "abc", "abc", "123"],
                "col2": ["abc", "abd", "abc", "abc", "123"],
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        feature = BaseCompareFeature(["col1", "col2"], ["col1", "col2"])
        feature._f_compare_vectorized = lambda s1_1, s1_2, s2_1, s2_2: np.ones(
            len(s1_1)
        )
        feature.compute(ix, A, B)


class TestCompareExact(TestData):
    """Test the exact comparison method."""

    def test_exact_str_type(self):
        A = DataFrame({"col": ["abc", "abc", "abc", "abc", "abc"]})
        B = DataFrame({"col": ["abc", "abd", "abc", "abc", "123"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        expected = DataFrame([1, 0, 1, 1, 0], index=ix)

        comp = recordlinkage.Compare()
        comp.exact("col", "col")
        result = comp.compute(ix, A, B)

        pdt.assert_frame_equal(result, expected)

    def test_exact_num_type(self):
        A = DataFrame({"col": [42, 42, 41, 43, nan]})
        B = DataFrame({"col": [42, 42, 42, 42, 42]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        expected = DataFrame([1, 1, 0, 0, 0], index=ix)

        comp = recordlinkage.Compare()
        comp.exact("col", "col")
        result = comp.compute(ix, A, B)

        pdt.assert_frame_equal(result, expected)

    def test_link_exact_missing(self):
        A = DataFrame({"col": ["a", "b", "c", "d", nan]})
        B = DataFrame({"col": ["a", "b", "d", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.exact("col", "col", label="na_")
        comp.exact("col", "col", missing_value=0, label="na_0")
        comp.exact("col", "col", missing_value=9, label="na_9")
        comp.exact("col", "col", missing_value=nan, label="na_na")
        comp.exact("col", "col", missing_value="str", label="na_str")
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series([1, 1, 0, 0, 0], index=ix, name="na_")
        pdt.assert_series_equal(result["na_"], expected)

        # Missing values as 0
        expected = Series([1, 1, 0, 0, 0], index=ix, name="na_0")
        pdt.assert_series_equal(result["na_0"], expected)

        # Missing values as 9
        expected = Series([1, 1, 0, 9, 9], index=ix, name="na_9")
        pdt.assert_series_equal(result["na_9"], expected)

        # Missing values as nan
        expected = Series([1, 1, 0, nan, nan], index=ix, name="na_na")
        pdt.assert_series_equal(result["na_na"], expected)

        # Missing values as string
        expected = Series([1, 1, 0, "str", "str"], index=ix, name="na_str")
        pdt.assert_series_equal(result["na_str"], expected)

    def test_link_exact_disagree(self):
        A = DataFrame({"col": ["a", "b", "c", "d", nan]})
        B = DataFrame({"col": ["a", "b", "d", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.exact("col", "col", label="d_")
        comp.exact("col", "col", disagree_value=0, label="d_0")
        comp.exact("col", "col", disagree_value=9, label="d_9")
        comp.exact("col", "col", disagree_value=nan, label="d_na")
        comp.exact("col", "col", disagree_value="str", label="d_str")
        result = comp.compute(ix, A, B)

        # disagree values as default
        expected = Series([1, 1, 0, 0, 0], index=ix, name="d_")
        pdt.assert_series_equal(result["d_"], expected)

        # disagree values as 0
        expected = Series([1, 1, 0, 0, 0], index=ix, name="d_0")
        pdt.assert_series_equal(result["d_0"], expected)

        # disagree values as 9
        expected = Series([1, 1, 9, 0, 0], index=ix, name="d_9")
        pdt.assert_series_equal(result["d_9"], expected)

        # disagree values as nan
        expected = Series([1, 1, nan, 0, 0], index=ix, name="d_na")
        pdt.assert_series_equal(result["d_na"], expected)

        # disagree values as string
        expected = Series([1, 1, "str", 0, 0], index=ix, name="d_str")
        pdt.assert_series_equal(result["d_str"], expected)


# tests/test_compare.py:TestCompareNumeric
class TestCompareNumeric(TestData):
    """Test the numeric comparison methods."""

    def test_numeric(self):
        A = DataFrame({"col": [1, 1, 1, nan, 0]})
        B = DataFrame({"col": [1, 2, 3, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric("col", "col", "step", offset=2)
        comp.numeric("col", "col", method="step", offset=2)
        comp.numeric("col", "col", "step", 2)
        result = comp.compute(ix, A, B)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=1)
        pdt.assert_series_equal(result[1], expected)

        # Basics
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=2)
        pdt.assert_series_equal(result[2], expected)

    def test_numeric_with_missings(self):
        A = DataFrame({"col": [1, 1, 1, nan, 0]})
        B = DataFrame({"col": [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric("col", "col", scale=2)
        comp.numeric("col", "col", scale=2, missing_value=0)
        comp.numeric("col", "col", scale=2, missing_value=123.45)
        comp.numeric("col", "col", scale=2, missing_value=nan)
        comp.numeric("col", "col", scale=2, missing_value="str")
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

        # Missing values as 0
        expected = Series([1.0, 1.0, 1.0, 0.0, 0.0], index=ix, dtype=np.float64, name=1)
        pdt.assert_series_equal(result[1], expected)

        # Missing values as 123.45
        expected = Series([1.0, 1.0, 1.0, 123.45, 123.45], index=ix, name=2)
        pdt.assert_series_equal(result[2], expected)

        # Missing values as nan
        expected = Series([1.0, 1.0, 1.0, nan, nan], index=ix, name=3)
        pdt.assert_series_equal(result[3], expected)

        # Missing values as string
        expected = Series([1, 1, 1, "str", "str"], index=ix, dtype=object, name=4)
        pdt.assert_series_equal(result[4], expected)

    @pytest.mark.parametrize("alg", NUMERIC_SIM_ALGORITHMS)
    def test_numeric_algorithms(self, alg):
        A = DataFrame({"col": [1, 1, 1, 1, 1]})
        B = DataFrame({"col": [1, 2, 3, 4, 5]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric("col", "col", method="step", offset=1, label="step")
        comp.numeric("col", "col", method="linear", offset=1, scale=2, label="linear")
        comp.numeric("col", "col", method="squared", offset=1, scale=2, label="squared")
        comp.numeric("col", "col", method="exp", offset=1, scale=2, label="exp")
        comp.numeric("col", "col", method="gauss", offset=1, scale=2, label="gauss")
        result_df = comp.compute(ix, A, B)

        result = result_df[alg]

        # All values between 0 and 1.
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

        if alg != "step":
            print(alg)
            print(result)

            # sim(scale) = 0.5
            expected_bool = Series(
                [False, False, False, True, False], index=ix, name=alg
            )
            pdt.assert_series_equal(result == 0.5, expected_bool)

            # sim(offset) = 1
            expected_bool = Series(
                [True, True, False, False, False], index=ix, name=alg
            )
            pdt.assert_series_equal(result == 1.0, expected_bool)

            # sim(scale) larger than 0.5
            expected_bool = Series(
                [False, False, True, False, False], index=ix, name=alg
            )
            pdt.assert_series_equal((result > 0.5) & (result < 1.0), expected_bool)

            # sim(scale) smaller than 0.5
            expected_bool = Series(
                [False, False, False, False, True], index=ix, name=alg
            )
            pdt.assert_series_equal((result < 0.5) & (result >= 0.0), expected_bool)

    @pytest.mark.parametrize("alg", NUMERIC_SIM_ALGORITHMS)
    def test_numeric_algorithms_errors(self, alg):
        # scale negative
        if alg != "step":
            with pytest.raises(ValueError):
                comp = recordlinkage.Compare()
                comp.numeric("age", "age", method=alg, offset=2, scale=-2)
                comp.compute(self.index_AB, self.A, self.B)

            # offset negative
            with pytest.raises(ValueError):
                comp = recordlinkage.Compare()
                comp.numeric("age", "age", method=alg, offset=-2, scale=-2)
                comp.compute(self.index_AB, self.A, self.B)

    def test_numeric_does_not_exist(self):
        # raise when algorithm doesn't exists

        A = DataFrame({"col": [1, 1, 1, nan, 0]})
        B = DataFrame({"col": [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.numeric("col", "col", method="unknown_algorithm")
        pytest.raises(ValueError, comp.compute, ix, A, B)


# tests/test_compare.py:TestCompareDates
class TestCompareDates(TestData):
    """Test the exact comparison method."""

    def test_dates(self):
        A = DataFrame(
            {
                "col": to_datetime(
                    ["2005/11/23", nan, "2004/11/23", "2010/01/10", "2010/10/30"]
                )
            }
        )
        B = DataFrame(
            {
                "col": to_datetime(
                    [
                        "2005/11/23",
                        "2010/12/31",
                        "2005/11/23",
                        "2010/10/01",
                        "2010/9/30",
                    ]
                )
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date("col", "col")
        result = comp.compute(ix, A, B)[0]

        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name=0)

        pdt.assert_series_equal(result, expected)

    def test_date_incorrect_dtype(self):
        A = DataFrame(
            {"col": ["2005/11/23", nan, "2004/11/23", "2010/01/10", "2010/10/30"]}
        )
        B = DataFrame(
            {
                "col": [
                    "2005/11/23",
                    "2010/12/31",
                    "2005/11/23",
                    "2010/10/01",
                    "2010/9/30",
                ]
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        A["col1"] = to_datetime(A["col"])
        B["col1"] = to_datetime(B["col"])

        comp = recordlinkage.Compare()
        comp.date("col", "col1")
        pytest.raises(ValueError, comp.compute, ix, A, B)

        comp = recordlinkage.Compare()
        comp.date("col1", "col")
        pytest.raises(ValueError, comp.compute, ix, A, B)

    def test_dates_with_missings(self):
        A = DataFrame(
            {
                "col": to_datetime(
                    ["2005/11/23", nan, "2004/11/23", "2010/01/10", "2010/10/30"]
                )
            }
        )
        B = DataFrame(
            {
                "col": to_datetime(
                    [
                        "2005/11/23",
                        "2010/12/31",
                        "2005/11/23",
                        "2010/10/01",
                        "2010/9/30",
                    ]
                )
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date("col", "col", label="m_")
        comp.date("col", "col", missing_value=0, label="m_0")
        comp.date("col", "col", missing_value=123.45, label="m_float")
        comp.date("col", "col", missing_value=nan, label="m_na")
        comp.date("col", "col", missing_value="str", label="m_str")
        result = comp.compute(ix, A, B)

        # Missing values as default
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name="m_")
        pdt.assert_series_equal(result["m_"], expected)

        # Missing values as 0
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name="m_0")
        pdt.assert_series_equal(result["m_0"], expected)

        # Missing values as 123.45
        expected = Series([1, 123.45, 0, 0.5, 0.5], index=ix, name="m_float")
        pdt.assert_series_equal(result["m_float"], expected)

        # Missing values as nan
        expected = Series([1, nan, 0, 0.5, 0.5], index=ix, name="m_na")
        pdt.assert_series_equal(result["m_na"], expected)

        # Missing values as string
        expected = Series([1, "str", 0, 0.5, 0.5], index=ix, dtype=object, name="m_str")
        pdt.assert_series_equal(result["m_str"], expected)

    def test_dates_with_swap(self):
        months_to_swap = [
            (9, 10, 123.45),
            (10, 9, 123.45),
            (1, 2, 123.45),
            (2, 1, 123.45),
        ]

        A = DataFrame(
            {
                "col": to_datetime(
                    ["2005/11/23", nan, "2004/11/23", "2010/01/10", "2010/10/30"]
                )
            }
        )
        B = DataFrame(
            {
                "col": to_datetime(
                    [
                        "2005/11/23",
                        "2010/12/31",
                        "2005/11/23",
                        "2010/10/01",
                        "2010/9/30",
                    ]
                )
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.date("col", "col", label="s_")
        comp.date("col", "col", swap_month_day=0, swap_months="default", label="s_1")
        comp.date(
            "col", "col", swap_month_day=123.45, swap_months="default", label="s_2"
        )
        comp.date(
            "col", "col", swap_month_day=123.45, swap_months=months_to_swap, label="s_3"
        )
        comp.date(
            "col",
            "col",
            swap_month_day=nan,
            swap_months="default",
            missing_value=nan,
            label="s_4",
        )
        comp.date("col", "col", swap_month_day="str", label="s_5")
        result = comp.compute(ix, A, B)

        # swap_month_day as default
        expected = Series([1, 0, 0, 0.5, 0.5], index=ix, name="s_")
        pdt.assert_series_equal(result["s_"], expected)

        # swap_month_day and swap_months as 0
        expected = Series([1, 0, 0, 0, 0.5], index=ix, name="s_1")
        pdt.assert_series_equal(result["s_1"], expected)

        # swap_month_day 123.45 (float)
        expected = Series([1, 0, 0, 123.45, 0.5], index=ix, name="s_2")
        pdt.assert_series_equal(result["s_2"], expected)

        # swap_month_day and swap_months 123.45 (float)
        expected = Series([1, 0, 0, 123.45, 123.45], index=ix, name="s_3")
        pdt.assert_series_equal(result["s_3"], expected)

        # swap_month_day and swap_months as nan
        expected = Series([1, nan, 0, nan, 0.5], index=ix, name="s_4")
        pdt.assert_series_equal(result["s_4"], expected)

        # swap_month_day as string
        expected = Series([1, 0, 0, "str", 0.5], index=ix, dtype=object, name="s_5")
        pdt.assert_series_equal(result["s_5"], expected)


# tests/test_compare.py:TestCompareGeo
class TestCompareGeo(TestData):
    """Test the geo comparison method."""

    def test_geo(self):
        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame(
            {
                "lat": [52.0842455, 52.3747388, 51.9280573],
                "lng": [5.0124516, 4.7585305, 4.4203581],
            }
        )
        B = DataFrame(
            {
                "lat": [52.3747388, 51.9280573, 52.0842455],
                "lng": [4.7585305, 4.4203581, 5.0124516],
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.geo("lat", "lng", "lat", "lng", method="step", offset=50)  # 50 km range
        result = comp.compute(ix, A, B)

        # Missing values as default [36.639460, 54.765854, 44.092472]
        expected = Series([1.0, 0.0, 1.0], index=ix, name=0)
        pdt.assert_series_equal(result[0], expected)

    def test_geo_batch(self):
        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame(
            {
                "lat": [52.0842455, 52.3747388, 51.9280573],
                "lng": [5.0124516, 4.7585305, 4.4203581],
            }
        )
        B = DataFrame(
            {
                "lat": [52.3747388, 51.9280573, 52.0842455],
                "lng": [4.7585305, 4.4203581, 5.0124516],
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.geo("lat", "lng", "lat", "lng", method="step", offset=1, label="step")
        comp.geo(
            "lat",
            "lng",
            "lat",
            "lng",
            method="linear",
            offset=1,
            scale=2,
            label="linear",
        )
        comp.geo(
            "lat",
            "lng",
            "lat",
            "lng",
            method="squared",
            offset=1,
            scale=2,
            label="squared",
        )
        comp.geo(
            "lat", "lng", "lat", "lng", method="exp", offset=1, scale=2, label="exp"
        )
        comp.geo(
            "lat", "lng", "lat", "lng", method="gauss", offset=1, scale=2, label="gauss"
        )
        result_df = comp.compute(ix, A, B)

        print(result_df)

        for alg in ["step", "linear", "squared", "exp", "gauss"]:
            result = result_df[alg]

            # All values between 0 and 1.
            assert (result >= 0.0).all()
            assert (result <= 1.0).all()

    def test_geo_does_not_exist(self):
        # Utrecht, Amsterdam, Rotterdam (Cities in The Netherlands)
        A = DataFrame(
            {
                "lat": [52.0842455, 52.3747388, 51.9280573],
                "lng": [5.0124516, 4.7585305, 4.4203581],
            }
        )
        B = DataFrame(
            {
                "lat": [52.3747388, 51.9280573, 52.0842455],
                "lng": [4.7585305, 4.4203581, 5.0124516],
            }
        )
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.geo("lat", "lng", "lat", "lng", method="unknown")
        pytest.raises(ValueError, comp.compute, ix, A, B)


class TestCompareStrings(TestData):
    """Test the exact comparison method."""

    def test_defaults(self):
        # default algorithm is levenshtein algorithm
        # test default values are indentical to levenshtein

        A = DataFrame({"col": ["str_abc", "str_abc", "str_abc", nan, "hsdkf"]})
        B = DataFrame({"col": ["str_abc", "str_abd", "jaskdfsd", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string("col", "col", label="default")
        comp.string("col", "col", method="levenshtein", label="with_args")
        result = comp.compute(ix, A, B)

        pdt.assert_series_equal(
            result["default"].rename(None), result["with_args"].rename(None)
        )

    def test_fuzzy(self):
        A = DataFrame({"col": ["str_abc", "str_abc", "str_abc", nan, "hsdkf"]})
        B = DataFrame({"col": ["str_abc", "str_abd", "jaskdfsd", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string("col", "col", method="jaro", missing_value=0)
        comp.string("col", "col", method="q_gram", missing_value=0)
        comp.string("col", "col", method="cosine", missing_value=0)
        comp.string("col", "col", method="jaro_winkler", missing_value=0)
        comp.string("col", "col", method="dameraulevenshtein", missing_value=0)
        comp.string("col", "col", method="levenshtein", missing_value=0)
        result = comp.compute(ix, A, B)

        print(result)

        assert result.notnull().all(1).all(0)
        assert (result[result.notnull()] >= 0).all(1).all(0)
        assert (result[result.notnull()] <= 1).all(1).all(0)

    def test_threshold(self):
        A = DataFrame({"col": ["gretzky", "gretzky99", "gretzky", "gretzky"]})
        B = DataFrame({"col": ["gretzky", "gretzky", nan, "wayne"]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string(
            "col",
            "col",
            method="levenshtein",
            threshold=0.5,
            missing_value=2.0,
            label="x_col1",
        )
        comp.string(
            "col",
            "col",
            method="levenshtein",
            threshold=1.0,
            missing_value=0.5,
            label="x_col2",
        )
        comp.string(
            "col",
            "col",
            method="levenshtein",
            threshold=0.0,
            missing_value=nan,
            label="x_col3",
        )
        comp.string(
            "col",
            "col",
            method="cosine",
            threshold=0.5,
            missing_value=nan,
            label="x_col4",
        )
        comp.string(
            "col",
            "col",
            method="q_gram",
            threshold=0.5,
            missing_value=nan,
            label="x_col5",
        )
        result = comp.compute(ix, A, B)

        expected = Series([1.0, 1.0, 2.0, 0.0], index=ix, name="x_col1")
        pdt.assert_series_equal(result["x_col1"], expected)

        expected = Series([1.0, 0.0, 0.5, 0.0], index=ix, name="x_col2")
        pdt.assert_series_equal(result["x_col2"], expected)

        expected = Series([1.0, 1.0, nan, 1.0], index=ix, name="x_col3")
        pdt.assert_series_equal(result["x_col3"], expected)

        expected = Series([1.0, 1.0, nan, 0.0], index=ix, name="x_col4")
        pdt.assert_series_equal(result["x_col4"], expected)

        expected = Series([1.0, 1.0, 0.0, 0.0], index=ix, name="x_col5")
        pdt.assert_series_equal(result["x_col5"], expected)

    @pytest.mark.parametrize("alg", STRING_SIM_ALGORITHMS)
    def test_incorrect_input(self, alg):
        A = DataFrame({"col": [1, 1, 1, nan, 0]})
        B = DataFrame({"col": [1, 1, 1, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        with pytest.raises((TypeError, AttributeError)):
            comp = recordlinkage.Compare()
            comp.string("col", "col", method=alg)
            comp.compute(ix, A, B)

    @pytest.mark.parametrize("alg", STRING_SIM_ALGORITHMS)
    def test_string_algorithms_nan(self, alg):
        A = DataFrame({"col": ["nan", nan, nan, nan, nan]})
        B = DataFrame({"col": ["nan", nan, nan, nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string("col", "col", method=alg)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, 0.0, 0.0, 0.0, 0.0], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

        comp = recordlinkage.Compare()
        comp.string("col", "col", method=alg, missing_value=nan)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, nan, nan, nan, nan], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

        comp = recordlinkage.Compare()
        comp.string("col", "col", method=alg, missing_value=9.0)
        result = comp.compute(ix, A, B)[0]

        expected = Series([1.0, 9.0, 9.0, 9.0, 9.0], index=ix, name=0)
        pdt.assert_series_equal(result, expected)

    @pytest.mark.parametrize("alg", STRING_SIM_ALGORITHMS)
    def test_string_algorithms(self, alg):
        A = DataFrame({"col": ["str_abc", "str_abc", "str_abc", nan, "hsdkf"]})
        B = DataFrame({"col": ["str_abc", "str_abd", "jaskdfsd", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string("col", "col", method=alg, missing_value=0)
        result = comp.compute(ix, A, B)[0]

        assert result.notnull().all()

        assert (result >= 0).all()
        assert (result <= 1).all()

        assert (result > 0).any()
        assert (result < 1).any()

    def test_fuzzy_does_not_exist(self):
        A = DataFrame({"col": ["str_abc", "str_abc", "str_abc", nan, "hsdkf"]})
        B = DataFrame({"col": ["str_abc", "str_abd", "jaskdfsd", nan, nan]})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        comp = recordlinkage.Compare()
        comp.string("col", "col", method="unknown_algorithm")
        pytest.raises(ValueError, comp.compute, ix, A, B)


class TestCompareFreq:
    def test_freq(self):
        # data
        array_repeated = np.repeat(np.arange(10), 10)
        array_tiled = np.tile(np.arange(20), 5)

        # convert to pandas data
        A = DataFrame({"col": array_repeated})
        B = DataFrame({"col": array_tiled})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # the part to test
        from recordlinkage.compare import Frequency
        from recordlinkage.compare import FrequencyA
        from recordlinkage.compare import FrequencyB

        comp = recordlinkage.Compare()
        comp.add(Frequency(left_on="col"))
        comp.add(FrequencyA("col"))
        result = comp.compute(ix, A, B)

        expected = Series(np.ones((100,)) / 10, index=ix)
        pdt.assert_series_equal(result[0], expected.rename(0))
        pdt.assert_series_equal(result[1], expected.rename(1))

        comp = recordlinkage.Compare()
        comp.add(Frequency(right_on="col"))
        comp.add(FrequencyB("col"))
        result = comp.compute(ix, A, B)

        expected = Series(np.ones((100,)) / 20, index=ix)
        pdt.assert_series_equal(result[0], expected.rename(0))
        pdt.assert_series_equal(result[1], expected.rename(1))

    def test_freq_normalise(self):
        # data
        array_repeated = np.repeat(np.arange(10), 10)
        array_tiled = np.tile(np.arange(20), 5)

        # convert to pandas data
        A = DataFrame({"col": array_repeated})
        B = DataFrame({"col": array_tiled})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # the part to test
        from recordlinkage.compare import Frequency

        comp = recordlinkage.Compare()
        comp.add(Frequency(left_on="col", normalise=False))
        result = comp.compute(ix, A, B)

        expected = DataFrame(np.ones((100,)) * 10, index=ix)
        pdt.assert_frame_equal(result, expected)

        comp = recordlinkage.Compare()
        comp.add(Frequency(right_on="col", normalise=False))
        result = comp.compute(ix, A, B)

        expected = DataFrame(np.ones((100,)) * 5, index=ix)
        pdt.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("missing_value", [0.0, np.nan, 10.0])
    def test_freq_nan(self, missing_value):
        # data
        array_repeated = np.repeat(np.arange(10, dtype=np.float64), 10)
        array_repeated[90:] = np.nan
        array_tiled = np.tile(np.arange(20, dtype=np.float64), 5)

        # convert to pandas data
        A = DataFrame({"col": array_repeated})
        B = DataFrame({"col": array_tiled})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # the part to test
        from recordlinkage.compare import Frequency

        comp = recordlinkage.Compare()
        comp.add(Frequency(left_on="col", missing_value=missing_value))
        result = comp.compute(ix, A, B)

        expected_np = np.ones((100,)) / 10
        expected_np[90:] = missing_value
        expected = DataFrame(expected_np, index=ix)
        pdt.assert_frame_equal(result, expected)


class TestCompareVariable:
    def test_variable(self):
        # data
        arrayA = np.random.random((100,))
        arrayB = np.random.random((100,))

        # convert to pandas data
        A = DataFrame({"col": arrayA})
        B = DataFrame({"col": arrayB})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # the part to test
        from recordlinkage.compare import Variable
        from recordlinkage.compare import VariableA
        from recordlinkage.compare import VariableB

        comp = recordlinkage.Compare()
        comp.add(Variable(left_on="col"))
        comp.add(VariableA("col"))
        result = comp.compute(ix, A, B)

        expected = Series(arrayA, index=ix)
        pdt.assert_series_equal(result[0], expected.rename(0))
        pdt.assert_series_equal(result[1], expected.rename(1))

        comp = recordlinkage.Compare()
        comp.add(Variable(right_on="col"))
        comp.add(VariableB("col"))
        result = comp.compute(ix, A, B)

        expected = Series(arrayB, index=ix)
        pdt.assert_series_equal(result[0], expected.rename(0))
        pdt.assert_series_equal(result[1], expected.rename(1))

    @pytest.mark.parametrize("missing_value", [0.0, np.nan, 10.0])
    def test_variable_nan(self, missing_value):
        # data
        arrayA = np.random.random((100,))
        arrayA[90:] = np.nan
        arrayB = np.random.random((100,))

        # convert to pandas data
        A = DataFrame({"col": arrayA})
        B = DataFrame({"col": arrayB})
        ix = MultiIndex.from_arrays([A.index.values, B.index.values])

        # the part to test
        from recordlinkage.compare import Variable

        comp = recordlinkage.Compare()
        comp.add(Variable(left_on="col", missing_value=missing_value))
        features = comp.compute(ix, A, B)
        result = features[0].rename(None)

        expected = Series(arrayA, index=ix)
        expected.iloc[90:] = missing_value
        pdt.assert_series_equal(result, expected)
