from __future__ import division
from __future__ import unicode_literals

import datetime
import unittest
import numpy as np
import recordlinkage.algorithms.conflict_resolution as cr


class TestConflictResolutionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = [datetime.datetime.now() for _ in range(10)]

    def test_remove(self):
        self.assertEqual(cr.remove_missing(((),), True, False),
                         ((),),
                         msg='Empty values, no metadata.')

        self.assertEqual(cr.remove_missing(((), ()), True, True),
                         ((), ()),
                         msg='Empty values and empty metadata.')

        self.assertEqual(cr.remove_missing(((1, 2, 3, 4),), True, False),
                         ((1, 2, 3, 4),),
                         msg='Values no metadata. No nans.')

        self.assertEqual(cr.remove_missing(((1, np.nan, 2, np.nan),), True, False),
                         ((1, 2),),
                         msg='Remove values, no metadata.')

        self.assertEqual(cr.remove_missing(((1, np.nan, 2, np.nan), (1, 2, 3, 4)), True, True),
                         ((1, 2), (1, 3)),
                         msg='Remove values, with metadata.')

        self.assertEqual(cr.remove_missing(((1, 2, 3, 4), (1, np.nan, 2, np.nan)), True, True),
                         ((1, 3), (1, 2)),
                         msg='Remove metadata, with values.')

        self.assertEqual(cr.remove_missing(((1, np.nan, 3, np.nan), (1, np.nan, 2, np.nan)), True, True),
                         ((1, 3), (1, 2)),
                         msg='Matching nans in values and metdata.')

        self.assertEqual(cr.remove_missing(((1, 3, np.nan, np.nan), (1, np.nan, 2, np.nan)), True, True),
                         ((1,), (1,)),
                         msg='Mismatching nans in values and metdata.')

        self.assertEqual(cr.remove_missing(((np.nan, 3, np.nan, np.nan), (1, np.nan, 2, np.nan)), True, True),
                         ((), ()),
                         msg='All nans')

        self.assertEqual(cr.remove_missing(((1, 3, np.nan, np.nan), (1, np.nan, 2, np.nan)), False, False),
                         ((1, 3, np.nan, np.nan), (1, np.nan, 2, np.nan)),
                         msg='Do not remove nans.')

        self.assertEqual(cr.remove_missing(((1, 3, np.nan, np.nan), (1, np.nan, 2, np.nan)), False, True),
                         ((1, np.nan), (1, 2)),
                         msg='Do not remove value nans.')

        self.assertEqual(
            cr.remove_missing(((self.times[0], 'aaa', np.nan, np.nan), (self.times[0], np.nan, 'bbb', np.nan)), False,
                              True),
            ((self.times[0], np.nan), (self.times[0], 'bbb')),
            msg='Non-numeric correctness.')

    def test_bool_getter(self):
        self.assertEqual(cr.bool_getter((True, True, False, False, True), lambda x: x)((0, 1, 2, 3, 4)),
                         (0, 1, 4),
                         msg='Basic correctness.')

        self.assertEqual(cr.bool_getter((False, False, False, False, False), lambda x: x)((0, 1, 2, 3, 4)),
                         (),
                         msg='None kept.')

        self.assertEqual(cr.bool_getter((True, True, True, True, True), lambda x: x)((0, 1, 2, 3, 4)),
                         (0, 1, 2, 3, 4),
                         msg='All kept.')

        self.assertEqual(
            cr.bool_getter((True, True, False, False, True), lambda x: x)(('0', ['1', {}], '2', '3', self.times[0])),
            ('0', ['1', {}], self.times[0]),
            msg='Non-numeric correctness.')

    def test_nullify(self):
        self.assertTrue(np.isnan(cr.nullify(((1, 2, 3), (4, 5, 6)), False)),
                        msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.nullify((('1', self.times[0], '3'), (self.times[1], '5', [[[6]]])), False)),
                        msg='Non-numeric correctness.')

    def test_choose_first(self):
        self.assertEqual(cr.choose_first(((1, 2, 3),), True),
                         1,
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_first(((),), True)),
                        msg='No data.')

        self.assertEqual(cr.choose_first(((np.nan, 1, 2),), True),
                         1,
                         msg='Ignore nan.')

        self.assertTrue(np.isnan(cr.choose_first(((np.nan, 1, 2),), False)),
                        msg='Do not ignore nan.')

        self.assertEqual(cr.choose_first((('a', self.times[0], [1, 2, 3]),), True),
                         'a',
                         msg='Non-numeric correctness.')

        self.assertEqual(cr.choose_first(((np.nan, self.times[0], [1, 2, 3]),), True),
                         self.times[0],
                         msg='Non-numeric correctness with nans.')

    def test_choose_last(self):
        self.assertEqual(cr.choose_last(((1, 2, 3),), True),
                         3,
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_last(((),), True)),
                        msg='No data.')

        self.assertEqual(cr.choose_last(((np.nan, 1, np.nan),), True),
                         1,
                         msg='Ignore nan.')

        self.assertTrue(np.isnan(cr.choose_last(((np.nan, 1, np.nan),), False)),
                        msg='Do not ignore nan.')

        self.assertEqual(cr.choose_last((('1', {2, }, self.times[0]),), True),
                         self.times[0],
                         msg='Non-numeric correctness.')

    def test_count(self):
        self.assertEqual(cr.count(((1, 1, 1, 2, 2, 3),), True),
                         3,
                         msg='Basic correctness.')

        self.assertEqual(cr.count(((1, 1, 1),), True),
                         1,
                         msg='One value')

        self.assertEqual(cr.count(((),), True),
                         0,
                         msg='No values')

        self.assertEqual(cr.count(((np.nan, 1, 2, np.nan),), True),
                         2,
                         msg='Remove nan.')

        self.assertEqual(cr.count(((np.nan, np.nan),), True),
                         0,
                         msg='All nans no values.')

        self.assertEqual(cr.count(((self.times[0], self.times[0], self.times[1], self.times[1], 'a', 'a'),), True),
                         3,
                         msg='Non-numeric correctness.')

    def test_choose_min(self):
        self.assertEqual(cr.choose_min(((4, 5, 2, 3, 4, 1),), True),
                         1,
                         msg='Basic correctness. Numeric.')

        self.assertEqual(cr.choose_min((('b', 'a', 'c'),), True),
                         'a',
                         msg='Basic correctness. String.')

        self.assertEqual(cr.choose_min(((self.times[0], self.times[1], self.times[2]),), True),
                         self.times[0],
                         msg='Basic correctness. Time.')

        self.assertTrue(np.isnan(cr.choose_min(((),), True)),
                        msg='No data.')

    def test_choose_max(self):
        self.assertEqual(cr.choose_max(((4, 5, 2, 3, 4, 1),), True),
                         5,
                         msg='Basic correctness. Numeric.')

        self.assertEqual(cr.choose_max((('b', 'a', 'c'),), True),
                         'c',
                         msg='Basic correctness. String.')

        self.assertEqual(cr.choose_max(((self.times[0], self.times[1], self.times[2]),), True),
                         self.times[2],
                         msg='Basic correctness. Time.')

        self.assertTrue(np.isnan(cr.choose_max(((),), True)),
                        msg='No data.')

    def test_choose_shortest(self):
        self.assertEqual(cr.choose_shortest((('abc', 'a', 'ab'),), cr.choose_min, True),
                         'a',
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_shortest(((),), cr.choose_min, True)),
                        msg='No data.')

        self.assertEqual(cr.choose_shortest((('abc', np.nan, 'ab'),), cr.choose_min, True),
                         'ab',
                         msg='Handles nan.')

        self.assertEqual(cr.choose_shortest((([1, 2], 'abcdef', self.times),), cr.choose_min, True),
                         [1, 2],
                         msg='Non-string correctness.')

        self.assertEqual(cr.choose_shortest((('bbb', 'aaa', 'ccc'),), cr.choose_min, True),
                         'aaa',
                         msg='Tie break min.')

        self.assertEqual(cr.choose_shortest((('bbb', 'aaa', 'ccc'),), cr.choose_first, True),
                         'bbb',
                         msg='Tie break first.')

        self.assertEqual(cr.choose_shortest((('bbb', 'aaa', 'ccc'),), cr.choose_last, True),
                         'ccc',
                         msg='Tie break last.')

        self.assertTrue(np.isnan(cr.choose_shortest((('bbb', 'aaa', 'ccc'),), cr.nullify, True)),
                        msg='Tie break null.')

    def test_choose_shortest_tie_break(self):
        self.assertEqual(cr.choose_shortest_tie_break((('abc', 'a', 'ab'),), True),
                         'a',
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_shortest_tie_break(((),), True)),
                        msg='No data.')

        self.assertEqual(cr.choose_shortest_tie_break((('abc', np.nan, 'ab'),), True),
                         'ab',
                         msg='Handles nan.')

        self.assertEqual(cr.choose_shortest_tie_break((([1, 2], 'abcdef', self.times),), True),
                         [1, 2],
                         msg='Non-string correctness.')

    def test_choose_longest(self):
        self.assertEqual(cr.choose_longest((('abc', 'a', 'ab'),), cr.choose_min, True),
                         'abc',
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_longest(((),), cr.choose_min, True)),
                        msg='No data.')

        self.assertEqual(cr.choose_longest((('abc', np.nan, 'ab'),), cr.choose_min, True),
                         'abc',
                         msg='Handles nan.')

        self.assertEqual(cr.choose_longest((([1, 2], 'abcdef', self.times),), cr.choose_min, True),
                         self.times,
                         msg='Non-string correctness.')

        self.assertEqual(cr.choose_longest((('bbb', 'aaa', 'ccc'),), cr.choose_min, True),
                         'aaa',
                         msg='Tie break min.')

        self.assertEqual(cr.choose_longest((('bbb', 'aaa', 'ccc'),), cr.choose_first, True),
                         'bbb',
                         msg='Tie break first.')

        self.assertEqual(cr.choose_longest((('bbb', 'aaa', 'ccc'),), cr.choose_last, True),
                         'ccc',
                         msg='Tie break last.')

        self.assertTrue(np.isnan(cr.choose_longest((('bbb', 'aaa', 'ccc'),), cr.nullify, True)),
                        msg='Tie break null.')

    def test_choose_longest_tie_break(self):
        self.assertEqual(cr.choose_longest_tie_break((('abc', 'a', 'ab'),), True),
                         'abc',
                         msg='Basic correctness.')

        self.assertTrue(np.isnan(cr.choose_longest_tie_break(((),), True)),
                        msg='No data.')

        self.assertEqual(cr.choose_longest_tie_break((('abc', np.nan, 'ab'),), True),
                         'abc',
                         msg='Handles nan.')

        self.assertEqual(cr.choose_longest_tie_break((([1, 2], 'abcdef', self.times),), True),
                         self.times,
                         msg='Non-string correctness.')

    def test_choose_random(self):
        self.assertIn(cr.choose_random(((1, 2, 3),), True),
                      (1, 2, 3),
                      msg='Basic correctness')

        self.assertIn(cr.choose_random((('a', 'b', 'c'),), True),
                      ('a', 'b', 'c'),
                      msg='String correctness.')

        self.assertTrue(np.isnan(cr.choose_random(((),), True)),
                        msg='No data.')

        self.assertTrue(np.isnan(cr.choose_random(((np.nan, np.nan, np.nan),), True)),
                        msg='Handles np.nan.')

        self.assertEqual(cr.choose_random(((np.nan, 1, np.nan),), True),
                         1,
                         msg='Handles np.nan.')

    def test_vote(self):
        self.assertEqual(cr.vote(((1, 1, 2, 2, 2, 3),), cr.choose_random, True),
                         2,
                         msg='Basic correctness.')

        self.assertEqual(
            cr.vote(((self.times[0], self.times[1], '2', '2', 2, 'three', 'three', 'three'),), cr.choose_random, True),
            'three',
            msg='Non-numeric correctness.')

        self.assertEqual(cr.vote(((1, 1, np.nan, 2, np.nan, 3),), cr.choose_random, True),
                         1,
                         msg='Handles nan.')

        self.assertEqual(cr.vote(((1, 1, 1, 3, 3, 3, 2, 2, 2),), cr.choose_first, True),
                         1,
                         msg='Tie break first.')

        self.assertEqual(cr.vote(((1, 1, 1, 3, 3, 3, 2, 2, 2),), cr.choose_last, True),
                         2,
                         msg='Tie break last.')

        self.assertEqual(cr.vote(((1, 1, 1, 3, 3, 3, 2, 2, 2),), cr.choose_max, True),
                         3,
                         msg='Tie break max.')

        self.assertEqual(cr.vote(((1, 1, 1, 3, 3, 3, 2, 2, 2),), cr.choose_min, True),
                         1,
                         msg='Tie break min.')

    def test_group(self):
        self.assertEqual(cr.group(((1, 2, 3),), 'set', True),
                         {1, 2, 3},
                         msg='Basic correctness with set.')

        self.assertEqual(cr.group(((1, 2, 3),), 'list', True),
                         [1, 2, 3],
                         msg='Basic correctness with list.')

        self.assertEqual(cr.group(((1, 2, 3),), 'tuple', True),
                         (1, 2, 3),
                         msg='Basic correctness with set.')

        self.assertEqual(cr.group(((1, np.nan, 3),), 'set', True),
                         {1, 3},
                         msg='Basic correctness with set and nan.')

        self.assertEqual(cr.group(((1, np.nan, 3),), 'list', True),
                         [1, 3],
                         msg='Basic correctness with list and nan.')

        self.assertEqual(cr.group(((1, np.nan, 3),), 'tuple', True),
                         (1, 3),
                         msg='Basic correctness with set and nan.')

        self.assertEqual(cr.group(((),), 'set', True),
                         set(),
                         msg='Basic correctness with set and nan.')

        self.assertEqual(cr.group(((),), 'list', True),
                         [],
                         msg='Basic correctness with list and nan.')

        self.assertEqual(cr.group(((),), 'tuple', True),
                         tuple(),
                         msg='Basic correctness with set and nan.')

    def test_no_gossip(self):
        self.assertEqual(cr.no_gossip(((1, 1, 1),), True),
                         1,
                         msg='Basic correctness.')

        self.assertEqual(cr.no_gossip((('a', 'a'),), True),
                         'a',
                         msg='Basic correctness string.')

        self.assertEqual(cr.no_gossip(((2, 2, np.nan, 2),), True),
                         2,
                         msg='Handles missing data.')

        self.assertTrue(np.isnan(cr.no_gossip(((2, 3),), True)),
                        msg='Inconsistent data correctness.')

        self.assertTrue(np.isnan(cr.no_gossip((('2', '3'),), True)),
                        msg='Inconsistent data correctness for string.')

    def test_aggregate(self):
        self.assertEqual(cr.aggregate(((1, 2, 3),), 'sum', True),
                         6,
                         msg='Sum correctness.')

        self.assertEqual(cr.aggregate(((1, 2, 3),), 'mean', True),
                         2,
                         msg='Mean correctness.')

        self.assertEqual(cr.aggregate(((1, 2, 3),), 'stdev', True),
                         np.std([1, 2, 3]),
                         msg='Stdev correctness.')

        self.assertEqual(cr.aggregate(((1, 2, 3),), 'var', True),
                         np.var([1, 2, 3]),
                         msg='Variance correctness.')

        self.assertEqual(cr.aggregate(((1, np.nan, 2, np.nan, 3),), 'sum', True),
                         6,
                         msg='Sum correctness with nan values.')

        self.assertEqual(cr.aggregate(((1, np.nan, 2, np.nan, 3),), 'mean', True),
                         2,
                         msg='Mean correctness with nan values.')

        self.assertEqual(cr.aggregate(((1, np.nan, 2, np.nan, 3),), 'stdev', True),
                         np.std([1, 2, 3]),
                         msg='Stdev correctness with nan values.')

        self.assertEqual(cr.aggregate(((1, np.nan, 2, np.nan, 3),), 'var', True),
                         np.var([1, 2, 3]),
                         msg='Variance correctness with nan values.')

        self.assertTrue(np.isnan(cr.aggregate(((1, np.nan, 2, np.nan, 3),), 'var', False)),
                        msg='Variance correctness with included nan values.')

    def test_choose_trusted(self):
        self.assertEqual(
            cr.choose_trusted(((1, 2), ('a', 'b')), 'a', cr.choose_random, cr.choose_random, True, True),
            1,
            msg='Basic correctness. Trust df_a.'
        )

        self.assertEqual(
            cr.choose_trusted(((1, 2), ('a', 'b')), 'b', cr.choose_random, cr.choose_random, True, True),
            2,
            msg='Basic correctness. Trust df_b.'
        )

        self.assertEqual(
            cr.choose_trusted(((1, 2, 3, 4), ('a', 'a', 'a', 'b')), 'a', cr.choose_first, cr.choose_random, True, True),
            1,
            msg='Break trusted tie with first.'
        )

        self.assertEqual(
            cr.choose_trusted(((1, 2, 3, 4), ('a', 'a', 'a', 'b')), 'a', cr.choose_max, cr.choose_random, True, True),
            3,
            msg='Break trusted tie with max.'
        )

        self.assertEqual(
            cr.choose_trusted(((1, 2, 3, 4), ('a', 'a', 'a', 'a')), 'b', cr.choose_random, cr.choose_last, True, True),
            4,
            msg='Break untrusted tie with last.'
        )

        self.assertEqual(
            cr.choose_trusted(((1, 2, 3, 4), ('a', 'a', 'a', 'a')), 'b', cr.choose_random, cr.choose_min, True, True),
            1,
            msg='Break untrusted tie with min.'
        )

    def test_annotated_concat(self):
        self.assertEqual(cr.annotated_concat(((1, 2, 3, 4, 5), ('a', 'b', 'c', 'd', 'e')), True, True),
                         [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')],
                         msg='Basic correctness.')

        self.assertEqual(cr.annotated_concat(((1, np.nan, 3, 4, 5), ('a', 'b', 'c', 'd', 'e')), True, True),
                         [(1, 'a'), (3, 'c'), (4, 'd'), (5, 'e')],
                         msg='nan in values.')

        self.assertEqual(cr.annotated_concat(((1, 2, 3, 4, 5), ('a', 'b', np.nan, 'd', 'e')), True, True),
                         [(1, 'a'), (2, 'b'), (4, 'd'), (5, 'e')],
                         msg='nan in metadata.')

        self.assertEqual(cr.annotated_concat(((), ()), True, True),
                         [],
                         msg='No data.')

    def test_choose_metadata_max(self):
        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd'), (2, 4, 3, 6)), cr.choose_random, True, True),
            'd',
            msg='Basic correctness.'
        )

        self.assertEqual(
            cr.choose_metadata_max(((np.nan, 'b', 'c', np.nan), (2, 4, 3, 6)), cr.choose_random, True, True),
            'b',
            msg='Handles nan values.'
        )

        self.assertEqual(
            cr.choose_metadata_max(((np.nan, 'b', np.nan, np.nan), (2, 4, 3, 6)), cr.choose_random, True, True),
            'b',
            msg='Handles nan values one choice.'
        )

        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd'), (2, np.nan, 3, np.nan)), cr.choose_random, True, True),
            'c',
            msg='Handles nan metadata.'
        )

        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd'), (np.nan, np.nan, 3, np.nan)), cr.choose_random, True, True),
            'c',
            msg='Handles nan metadata one choice.'
        )

        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd', 'e'), (2, 4, 4, 2, 4)), cr.choose_first, True, True),
            'b',
            msg='Break tie with first.'
        )

        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd', 'e'), (2, 4, 4, 2, 4)), cr.choose_last, True, True),
            'e',
            msg='Break tie with last.'
        )

        self.assertEqual(
            cr.choose_metadata_max((('a', 'b', 'c', 'd'), (self.times[2], self.times[4], self.times[3], self.times[6])),
                                   cr.choose_random, True, True),
            'd',
            msg='Correctness with datetime metadata.'
        )

    def test_choose_metadata_min(self):
        self.assertEqual(
            cr.choose_metadata_min((('a', 'b', 'c', 'd'), (2, 4, 3, 6)), cr.choose_random, True, True),
            'a',
            msg='Basic correctness.'
        )

        self.assertEqual(
            cr.choose_metadata_min(((np.nan, 'b', 'c', np.nan), (2, 4, 3, 6)), cr.choose_random, True, True),
            'c',
            msg='Handles nan values.'
        )

        self.assertEqual(
            cr.choose_metadata_min((('a', 'b', 'c', 'd'), (2, np.nan, 3, np.nan)), cr.choose_random, True, True),
            'a',
            msg='Handles nan metadata.'
        )

        self.assertEqual(
            cr.choose_metadata_min((('a', 'b', 'c', 'd', 'e'), (2, 4, 4, 2, 4)), cr.choose_first, True, True),
            'a',
            msg='Break tie with first.'
        )

        self.assertEqual(
            cr.choose_metadata_min((('a', 'b', 'c', 'd', 'e'), (2, 4, 4, 2, 4)), cr.choose_last, True, True),
            'd',
            msg='Break tie with last.'
        )

        self.assertEqual(
            cr.choose_metadata_min((('a', 'b', 'c', 'd'), (self.times[2], self.times[4], self.times[3], self.times[6])),
                                   cr.choose_random, True, True),
            'a',
            msg='Correctness with datetime metadata.'
        )
