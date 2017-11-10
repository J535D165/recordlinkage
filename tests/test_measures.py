from __future__ import division

import unittest
import recordlinkage as rl

import numpy
import pandas

FULL_INDEX = pandas.MultiIndex.from_product(
    [[1, 2, 3], [1, 2, 3]],  # 3x3 matrix
    names=['first', 'second']
)
LINKS_TRUE = pandas.MultiIndex.from_tuples(
    [(1, 1), (2, 2), (3, 3)],  # the diagonal
    names=['first', 'second']
)
LINKS_PRED = pandas.MultiIndex.from_tuples(
    [(1, 1), (2, 1), (3, 1), (1, 2)],  # L shape
    names=['first', 'second']
)


class TestMeasures(unittest.TestCase):

    def test_confusion_matrix(self):

        result = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        expected = numpy.array([[1, 2], [3, 3]])

        numpy.testing.assert_array_equal(result, expected)

    def test_tp_fp_tn_fn(self):

        tp = rl.true_positives(LINKS_TRUE, LINKS_PRED)
        self.assertEqual(tp, 1)
        fp = rl.false_positives(LINKS_TRUE, LINKS_PRED)
        self.assertEqual(fp, 3)
        tn = rl.true_negatives(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        self.assertEqual(tn, 3)
        fn = rl.false_negatives(LINKS_TRUE, LINKS_PRED)
        self.assertEqual(fn, 2)

    def test_recall(self):

        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED)

        self.assertEqual(rl.recall(LINKS_TRUE, LINKS_PRED), 1 / 3)
        self.assertEqual(rl.recall(cm), 1 / 3)

    def test_precision(self):

        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        self.assertEqual(rl.precision(LINKS_TRUE, LINKS_PRED), 1 / 4)
        self.assertEqual(rl.precision(cm), 1 / 4)

    def test_accuracy(self):

        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        self.assertEqual(rl.accuracy(
            LINKS_TRUE, LINKS_PRED, len(FULL_INDEX)), 4 / 9)
        self.assertEqual(rl.accuracy(cm), 4 / 9)

    def test_specificity(self):

        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))

        self.assertEqual(rl.specificity(
            LINKS_TRUE, LINKS_PRED, len(FULL_INDEX)), 1 / 2)
        self.assertEqual(rl.specificity(cm), 1 / 2)

    def test_fscore(self):

        # confusion matrix
        cm = rl.confusion_matrix(LINKS_TRUE, LINKS_PRED, len(FULL_INDEX))
        prec = rl.precision(LINKS_TRUE, LINKS_PRED)
        rec = rl.recall(LINKS_TRUE, LINKS_PRED)
        expected = float(2 * prec * rec / (prec + rec))

        self.assertEqual(rl.fscore(LINKS_TRUE, LINKS_PRED), expected)
        self.assertEqual(rl.fscore(cm), expected)

    def test_full_index_size(self):

        df_a = pandas.DataFrame(numpy.arange(10))
        df_b = pandas.DataFrame(numpy.arange(10))

        self.assertEqual(rl.full_index_size(df_a), 45)
        self.assertEqual(rl.full_index_size(len(df_a)), 45)
        self.assertEqual(rl.full_index_size((len(df_a))), 45)
        self.assertEqual(rl.full_index_size([len(df_a)]), 45)

        self.assertEqual(rl.full_index_size(df_a, df_b), 100)
        self.assertEqual(rl.full_index_size(len(df_a), len(df_b)), 100)
        self.assertEqual(rl.full_index_size((len(df_a), len(df_b))), 100)
        self.assertEqual(rl.full_index_size([len(df_a), len(df_b)]), 100)

    def test_reduction_ratio(self):

        df_a = pandas.DataFrame(numpy.arange(10))
        df_b = pandas.DataFrame(numpy.arange(10))
        candidate_pairs_link = pandas.MultiIndex.from_product(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        )
        candidate_pairs_dedup = pandas.MultiIndex.from_arrays(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        )

        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_dedup, df_a), 8 / 9
        )
        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_dedup, (df_a)), 8 / 9
        )
        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_dedup, (df_a,)), 8 / 9
        )

        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_link, df_a, df_b), 3 / 4
        )
        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_link, (df_a, df_b)), 3 / 4
        )
        self.assertEqual(
            rl.reduction_ratio(candidate_pairs_link, [df_a, df_b]), 3 / 4
        )
