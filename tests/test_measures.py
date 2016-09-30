import unittest
import recordlinkage

import numpy
import pandas

pairs = [(i, j) for i in range(0, 10) for j in range(0, 10)]
true_matches = [(1, 2), (2, 2), (2, 1), (3, 2), (5, 2),
                (4, 9), (5, 6), (6, 7), (1, 6)]
matches = [(1, 2), (2, 2), (2, 1), (5, 2), (4, 9),
           (1, 6), (9, 9), (8, 8), (0, 8)]

index = pandas.MultiIndex.from_tuples(pairs, names=['first', 'second'])
gold_matches_index = pandas.MultiIndex.from_tuples(
    true_matches, names=['first', 'second'])
matches_index = pandas.MultiIndex.from_tuples(
    matches, names=['first', 'second'])

CONF_M1 = numpy.array([[100, 0], [0, 1000]])
CONF_M2 = numpy.array([[100, 10], [20, 1000]])
CONF_M3 = numpy.array([[100, 0], [20, 1000]])
CONF_M4 = numpy.array([[100, 10], [0, 1000]])
CONF_M5 = numpy.array([[0, 10], [10, 0]])


class TestMeasures(unittest.TestCase):

    def test_confusion_matrix(self):

        conf_m = recordlinkage.confusion_matrix(
            gold_matches_index, matches_index, len(index))
        conf_m_2 = recordlinkage.confusion_matrix(
            gold_matches_index, matches_index, index)

        self.assertEqual(numpy.sum(conf_m), len(pairs))
        self.assertEqual(numpy.sum(conf_m[0, :]), len(gold_matches_index))
        self.assertEqual(numpy.sum(conf_m[:, 0]), len(matches_index))

        numpy.testing.assert_array_equal(conf_m, conf_m_2)

    def test_recall(self):

        self.assertEqual(recordlinkage.recall(CONF_M1), 1.0)
        self.assertEqual(recordlinkage.recall(CONF_M5), 0.0)

    def test_precision(self):

        self.assertEqual(recordlinkage.precision(CONF_M1), 1.0)
        self.assertEqual(recordlinkage.precision(CONF_M5), 0.0)

    def test_accuracy(self):

        self.assertEqual(recordlinkage.accuracy(CONF_M1), 1.0)
        self.assertEqual(recordlinkage.accuracy(CONF_M5), 0.0)

    def test_specificity(self):

        self.assertEqual(recordlinkage.specificity(CONF_M1), 1.0)
        self.assertEqual(recordlinkage.specificity(CONF_M5), 0.0)

    def test_fscore(self):

        self.assertEqual(recordlinkage.fscore(CONF_M1), 1.0)
        self.assertRaises(ZeroDivisionError, recordlinkage.fscore, CONF_M5)
