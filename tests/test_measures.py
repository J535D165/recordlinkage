#!/usr/bin/env python


import numpy
import pandas

import recordlinkage as rl

FULL_INDEX = pandas.MultiIndex.from_product(
    [[1, 2, 3], [1, 2, 3]], names=["first", "second"]  # 3x3 matrix
)
LINKS_TRUE = pandas.MultiIndex.from_tuples(
    [(1, 1), (2, 2), (3, 3)], names=["first", "second"]  # the diagonal
)
LINKS_PRED = pandas.MultiIndex.from_tuples(
    [(1, 1), (2, 1), (3, 1), (1, 2)], names=["first", "second"]  # L shape
)


class TestMeasures:
    def test_confusion_matrix(self):
        result_len = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        result_full_index = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, FULL_INDEX)
        expected = numpy.array([[1, 2], [3, 3]])

        numpy.testing.assert_array_equal(result_len, expected)
        numpy.testing.assert_array_equal(result_full_index, expected)

    def test_tp_fp_tn_fn(self):
        tp = rl.true_positives(LINKS_TRUE, LINKS_PRED)
        assert tp == 1
        fp = rl.false_positives(LINKS_TRUE, LINKS_PRED)
        assert fp == 3
        tn = rl.true_negatives(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        assert tn == 3
        fn = rl.false_negatives(LINKS_TRUE, LINKS_PRED)
        assert fn == 2

    def test_recall(self):
        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED)

        assert rl.recall(LINKS_TRUE, LINKS_PRED) == 1 / 3
        assert rl.recall(cm) == 1 / 3

    def test_precision(self):
        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        assert rl.precision(LINKS_TRUE, LINKS_PRED) == 1 / 4
        assert rl.precision(cm) == 1 / 4

    def test_accuracy(self):
        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        assert rl.accuracy(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX)) == 4 / 9
        assert rl.accuracy(cm) == 4 / 9
        assert rl.accuracy(LINKS_TRUE, LINKS_PRED, FULL_INDEX) == 4 / 9

    def test_specificity(self):
        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        assert rl.specificity(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX)) == 1 / 2
        assert rl.specificity(cm) == 1 / 2
        assert rl.specificity(LINKS_TRUE, LINKS_PRED, FULL_INDEX) == 1 / 2

    def test_fscore(self):
        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        prec = rl.precision(LINKS_TRUE, LINKS_PRED)
        rec = rl.recall(LINKS_TRUE, LINKS_PRED)
        expected = float(2 * prec * rec / (prec + rec))

        assert rl.fscore(LINKS_TRUE, LINKS_PRED) == expected
        assert rl.fscore(cm) == expected

    def test_full_index_size(self):
        df_a = pandas.DataFrame(numpy.arange(10))
        df_b = pandas.DataFrame(numpy.arange(10))

        assert rl.full_index_size(df_a) == 45
        assert rl.full_index_size(len(df_a)) == 45
        assert rl.full_index_size(len(df_a)) == 45
        assert rl.full_index_size([len(df_a)]) == 45

        assert rl.full_index_size(df_a, df_b) == 100
        assert rl.full_index_size(len(df_a), len(df_b)) == 100
        assert rl.full_index_size((len(df_a), len(df_b))) == 100
        assert rl.full_index_size([len(df_a), len(df_b)]) == 100

    def test_reduction_ratio(self):
        df_a = pandas.DataFrame(numpy.arange(10))
        df_b = pandas.DataFrame(numpy.arange(10))
        candidate_pairs_link = pandas.MultiIndex.from_product(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        )
        candidate_pairs_dedup = pandas.MultiIndex.from_arrays(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        )

        assert rl.reduction_ratio(candidate_pairs_dedup, df_a) == 8 / 9
        assert rl.reduction_ratio(candidate_pairs_dedup, (df_a)) == 8 / 9
        assert rl.reduction_ratio(candidate_pairs_dedup, (df_a,)) == 8 / 9

        assert rl.reduction_ratio(candidate_pairs_link, df_a, df_b) == 3 / 4
        assert rl.reduction_ratio(candidate_pairs_link, (df_a, df_b)) == 3 / 4
        assert rl.reduction_ratio(candidate_pairs_link, [df_a, df_b]) == 3 / 4
