from __future__ import division
from __future__ import unicode_literals

import unittest
import datetime
import warnings
import functools
import itertools as it
import multiprocessing as mp
from six import string_types
from random import randrange
from datetime import timedelta
from parameterized import parameterized, param

import pandas
import numpy as np
from numpy import nan, arange

import recordlinkage
from recordlinkage.fusion import FuseDuplicates
import recordlinkage.algorithms.conflict_resolution as cr


def validate_job(job):
    def if_not_none(key, f, *args, **kwargs):
        return f(job[key], *args, **kwargs) if job[key] is not None else True

    def warn_on_fail(cond, msg):
        if not cond:
            raise Warning(msg)
        return cond

    handler = job['handler'].__name__

    checks = [
        # The job
        warn_on_fail(isinstance(job, dict), 'job not dict'),
        # CR Function
        warn_on_fail('fun' in job.keys(), 'no resolution function'),
        warn_on_fail(callable(job['fun']) if handler != '_do_keep' else True,
                     ' resolution function ({}) not callable'.format(job['fun'])),
        # Handling function
        warn_on_fail('handler' in job.keys(), 'no handling method'),
        warn_on_fail(callable(job['handler']), 'handling method not callable'),
        # Data columns
        warn_on_fail(job['values_a'] is not None if handler != '_do_keep' else True,
                     'values_a is None when not keeping original columns'),
        warn_on_fail(job['values_b'] is not None if handler != '_do_keep' else True,
                     'values_b is None when not keeping original columns'),
        warn_on_fail(job['values_a'] is not None or job['values_b'] is not None if handler != '_do_keep' else True,
                     'both values_a and values_b are None'),
        warn_on_fail(if_not_none('values_a', isinstance, (list, ) + string_types), 'bad values_a type'),
        warn_on_fail(if_not_none('values_b', isinstance, (list, ) + string_types), 'bad vlaues_b type'),
        # Metadata columns
        warn_on_fail(if_not_none('meta_a', isinstance, (list, ) + string_types), 'bad meta_a type'),
        warn_on_fail(if_not_none('meta_b', isinstance, (list, ) + string_types), 'bad meta_b type'),
        # Transformation functions
        warn_on_fail(if_not_none('transform_vals', callable), 'transform_vals not callable'),
        warn_on_fail(if_not_none('transform_meta', callable), 'transform_meta not callable'),
        # Args and Kwargs
        warn_on_fail(if_not_none('params', isinstance, tuple), 'bad params type'),
        warn_on_fail(if_not_none('kwargs', isinstance, dict), 'bad kwargs type'),
        # Name
        warn_on_fail(if_not_none('name', isinstance, string_types), 'bad name type'),
        # Description
        warn_on_fail(if_not_none('description', isinstance, string_types), 'bad description type')
    ]
    result = functools.reduce(lambda a, b: a and b, checks)
    warn_on_fail(result, 'job failed:\n' + str(job))
    return result


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.

    Taken from: https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


# Inluded test data as test_compare.py
FIRST_NAMES = [u'Ronald', u'Amy', u'Andrew', u'William', u'Frank', u'Jessica',
               u'Kevin', u'Tyler', u'Yvonne', nan]
LAST_NAMES = [u'Graham', u'Smith', u'Holt', u'Pope', u'Hernandez',
              u'Gutierrez', u'Rivera', nan, u'Crane', u'Padilla']
STREET = [u'Oliver Neck', nan, u'Melissa Way', u'Sara Dale',
          u'Keith Green', u'Olivia Terrace', u'Williams Trail',
          u'Durham Mountains', u'Anna Circle', u'Michelle Squares']
JOB = [u'Designer, multimedia', u'Designer, blown glass/stained glass',
       u'Chiropractor', u'Engineer, mining', u'Quantity surveyor',
       u'Phytotherapist', u'Teacher, English as a foreign language',
       u'Electrical engineer', u'Research officer, government', u'Economist']

# Add multiple numeric fields for meet_in_the_middle
AGES = [23, 40, 70, 45, 23, 57, 38, nan, 45, 46]
INCOMES = [23000, 40000, nan, 70000, 45000, 23000, 57000, 38000, 45000, 46000]

# Add time metadata
DATES = [random_date(datetime.datetime(2016, 1, 1), datetime.datetime(2017, 1, 1))]

# Combinations of columns to resolve on
STR_COLNAMES = ['given_name', ['given_name', 'last_name']]

# Use multiprocessing?
MP_OPTION = [1, mp.cpu_count()]

# Cases for strategies without type requirements
GENERAL_STATEGIES = [
    {'method': 'no_gossiping', 'params': [], 'kws': {}},
    {'method': 'roll_the_dice', 'params': [], 'kws': {}},
    {'method': 'pass_it_on', 'params': [], 'kws': {'kind': 'tuple'}},
    {'method': 'pass_it_on', 'params': [], 'kws': {'kind': 'set'}},
    {'method': 'pass_it_on', 'params': [], 'kws': {'kind': 'list'}},
    {'method': 'trust_your_friends', 'params': [], 'kws': {'trusted': 'a'}},
    {'method': 'trust_your_friends', 'params': [], 'kws': {'trusted': 'b'}},
    {'method': 'keep_original', 'params': [], 'kws': {}}
]

# Tie breaking conflict resolution functions
TIE_BREAK_OPTIONS = [
    'random', 'trust_a', 'trust_b',
    'min', 'max', 'shortest',
    'longest', 'null'
]

# Cases that require tie breaks
TIE_BREAK_STRATEGIES = [
    {'method': 'cry_with_the_wolves', 'params': [], 'kws': {}},
    {'method': 'keep_up_to_date', 'params': ['date', 'date'], 'kws': {}},
    {'method': 'choose_by_scored_value', 'params': [hash, ], 'kws': {}},
    {'method': 'choose_by_scored_metadata', 'params': ['given_name', 'given_name', hash, ], 'kws': {}}
]

# Combinations of columns to resolve on
NUM_COLNAMES = ['income', 'age', ['income', 'age']]

# Numeric aggregation options
NUMERIC_OPTIONS = [
    'sum', 'mean', 'stdev', 'var'
]

# Cases that use numeric aggregation
NUMERIC_STRATEGIES = [
    {'method': 'meet_in_the_middle', 'params': [], 'kws': {}},
]

RESOLUTION_CASES = it.chain(
    [param(case[0]['method'], case[3], case[1], case[2], *case[0]['params'], **case[0]['kws'])
     for case in it.product(GENERAL_STATEGIES, STR_COLNAMES, STR_COLNAMES, MP_OPTION)],
    [param(case[0]['method'], case[4], case[1], case[2], *case[0]['params'], tie_break=case[3], **case[0]['kws'])
     for case in it.product(TIE_BREAK_STRATEGIES, STR_COLNAMES, STR_COLNAMES, TIE_BREAK_OPTIONS, MP_OPTION)],
    [param(case[0]['method'], case[4], case[1], case[2], case[3], *case[0]['params'], **case[0]['kws'])
     for case in it.product(NUMERIC_STRATEGIES, NUM_COLNAMES, NUM_COLNAMES, NUMERIC_OPTIONS, MP_OPTION)],
)


class TestFuseLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N_A = 100
        N_B = 100

        def str_nan(s):
            return np.nan if s == 'nan' else s

        cls.A = pandas.DataFrame({
            'age': pandas.Series(np.random.choice(AGES, N_A)),
            'given_name': pandas.Series(np.random.choice(FIRST_NAMES, N_A)).apply(str_nan),
            'last_name': pandas.Series(np.random.choice(LAST_NAMES, N_A)).apply(str_nan),
            'street': pandas.Series(np.random.choice(STREET, N_A)).apply(str_nan),
            'income': pandas.Series(np.random.choice(INCOMES, N_A)),
            'date': pandas.Series(np.random.choice(DATES, N_A))
        })

        cls.B = pandas.DataFrame({
            'age': pandas.Series(np.random.choice(AGES, N_B)),
            'given_name': pandas.Series(np.random.choice(FIRST_NAMES, N_B)).apply(str_nan),
            'last_name': pandas.Series(np.random.choice(LAST_NAMES, N_B)).apply(str_nan),
            'street': pandas.Series(np.random.choice(STREET, N_B)).apply(str_nan),
            'income': pandas.Series(np.random.choice(INCOMES, N_B)),
            'date': pandas.Series(np.random.choice(DATES, N_B))
        })

        cls.A.index.name = 'index_df1'
        cls.B.index.name = 'index_df2'

        cls.index_AB = pandas.MultiIndex.from_arrays(
            [arange(len(cls.A)), arange(len(cls.B))],
            names=[cls.A.index.name, cls.B.index.name])

        cls.counter = 0

    def setUp(self):
        self.comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        self.fuse = recordlinkage.FuseLinks()
        pass

    # Test takes long time
    @parameterized.expand(RESOLUTION_CASES)
    def test_resolution_result(self, method_to_call, mp_option, *args, **kwargs):
        """Validate job metadata and check conflict resolution result is a pandas series."""
        self.fuse.keep_original(args[0], args[1])
        getattr(self.fuse, method_to_call)(*args, **kwargs)

        self.assertTrue(validate_job(self.fuse.resolution_queue[-1]), 'resolution queue job failed validation')
        # Check job runs and produces dataframe.
        result = self.fuse.fuse(self.comp.vectors.index, self.A, self.B, njobs=mp_option)
        self.assertIsInstance(result, pandas.DataFrame, 'result not a dataframe')

    # Test takes long time
    def test_job_naming_error(self):
        for _ in range(2000):
            self.fuse.roll_the_dice('age', 'age', name='conflicting_name')
        with self.assertRaises(RuntimeError):
            self.fuse._resolve_job_names('_')

    def test_keep_original_suffix(self):
        """Test suffix overrid in FuseLinks.keep_original."""
        self.fuse.keep_original(['age'], ['given_name'], 'from_a', 'from_b')
        # Validate the job metadata
        self.assertTrue(validate_job(self.fuse.resolution_queue[0]), 'resolution queue job failed validation')

        # Check job runs and produces dataframe.
        result = self.fuse.fuse(self.comp.vectors.index, self.A, self.B, njobs=1)
        self.assertIsInstance(result, pandas.DataFrame, 'result not a dataframe')

    def test_keep_original_errors(self):
        self.fuse.keep_original(['age'], ['given_name'])
        # Invalidate metadata
        self.fuse.resolution_queue[0]['values_a'] = ['age', 'age']
        self.fuse.resolution_queue[0]['values_b'] = ['age', 'age']
        # Check job runs and produces dataframe.
        with self.assertRaises(AssertionError):
            self.fuse.fuse(self.comp.vectors.index, self.A, self.B, njobs=1)

    def test_data_generalization(self):
        """Test that FuseLinks._make_resolution_series generalizes values and metadata when appropriate"""
        self.fuse._fusion_init(self.comp.vectors.index, self.comp.df_a, self.comp.df_b, None, None)
        result_1 = self.fuse._make_resolution_series(
            ['age'], ['age'], ['given_name', 'given_name'], ['given_name', 'given_name']
        )
        result_2 = self.fuse._make_resolution_series(
            ['given_name', 'given_name'], ['given_name', 'given_name'], ['age'], ['age']
        )
        for t in result_1:
            self.assertTrue(len(t[0]) == len(t[1]), 'FuseLinks failed to generalize values to metadata.')
        for t in result_2:
            self.assertTrue(len(t[0]) == len(t[1]), 'FuseLinks failed to generalized metadata to values.')

    def test_resolution_series_assertions(self):
        """Test that FuseLinks._make_resolution_series generalizes values and metadata when appropriate"""
        with self.assertRaises(AssertionError):
            # No data a
            self.fuse._make_resolution_series(
                ['age'], ['age'], ['given_name', 'given_name'], ['given_name', 'given_name']
            )

        with self.assertRaises(AssertionError):
            # No data b
            self.fuse.df_a = pandas.DataFrame()
            self.fuse._make_resolution_series(
                ['age'], ['age'], ['given_name', 'given_name'], ['given_name', 'given_name']
            )

        # Check post-initialization assertions.
        self.fuse._fusion_init(self.comp.vectors.index, self.comp.df_a, self.comp.df_b, None, None)
        with self.assertRaises(ValueError):
            # Bad transformation function
            self.fuse._make_resolution_series(
                ['age'], ['age'], ['given_name', 'given_name'], ['given_name', 'given_name'], transform_vals=1
            )
        with self.assertRaises(ValueError):
            # Bad transformation function
            self.fuse._make_resolution_series(
                ['age'], ['age'], ['given_name', 'given_name'], ['given_name', 'given_name'], transform_meta=1
            )

    def test_bad_tie_break_value(self):
        with self.assertRaises(ValueError):
            self.fuse.keep_up_to_date('age', 'age', 'date', 'date', tie_break='invalid strategy')
        with self.assertRaises(ValueError):
            self.fuse.keep_up_to_date('age', 'age', 'date', 'date', tie_break=42)
        with self.assertRaises(ValueError):
            self.fuse.keep_up_to_date('age', 'age', 'date', 'date', tie_break=None)

    def test_process_tie_break_function(self):
        self.fuse.keep_up_to_date('age', 'age', 'date', 'date', tie_break=cr.choose_random)

    def test_resolve_typing(self):
        self.fuse.resolve(cr.choose_random, 'age', 'age')

    def test_use_predictions(self):
        self.fuse.keep_original('age', [])
        # Make arbitrary classification series
        preds = pandas.Series([(i % 5) == 0 for i in range(len(self.comp.vectors.index))])
        fused = self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b, predictions=preds)
        count = sum(1 if b else 0 for b in preds)
        self.assertEqual(len(fused), count, msg='Length of fused output incorrect after prediction application.')

    def test_prediction_type(self):
        self.fuse.keep_original('age', [])
        # Make arbitrary classification series
        preds = pandas.Series([(i % 5) == 0 for i in range(len(self.comp.vectors.index))])
        with self.assertRaises(ValueError):
            fused = self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b, predictions=1)

    def test_job_naming_correctness(self):
        for _ in range(5):
            self.fuse.roll_the_dice('age', 'age', name='conflicting_name')
            self.fuse.roll_the_dice('age', 'age', name='another_name')
        fused = self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b)
        self.assertListEqual(
            list(fused.columns),
            ['conflicting_name', 'another_name',
             'conflicting_name_1', 'another_name_1',
             'conflicting_name_2', 'another_name_2',
             'conflicting_name_3', 'another_name_3',
             'conflicting_name_4', 'another_name_4']
        )

    def test_exceeds_core_limit(self):
        self.fuse.roll_the_dice('age', 'age')
        with self.assertWarns(RuntimeWarning):
            self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b, njobs=100)

    def test_fusedups_not_implemented(self):
        with self.assertWarns(UserWarning):
            fuse_d = FuseDuplicates()
        with self.assertWarns(UserWarning):
            self.assertIsInstance(
                fuse_d._find_clusters('a'),
                type(NotImplemented)
            )
        with self.assertWarns(UserWarning):
            self.assertIsInstance(
                fuse_d._make_resolution_series('a', 'b'),
                type(NotImplemented)
            )

    def test_conflict_resolution_transform_vals(self):
        def static_test_fun(x):
            return True

        self.fuse.keep_original('age', 'age')
        self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b)
        data = self.fuse._make_resolution_series(
            'age', 'age',
            meta_a='age', meta_b='age',
            transform_vals=static_test_fun,
            transform_meta=static_test_fun
        )
        for i in data:
            self.assertTupleEqual(i, ((True, True), (True, True)))

    def test_resolution_series_insufficient_metadata(self):
        self.fuse.keep_original('age', 'age')
        self.fuse.fuse(self.comp.vectors.index, self.comp.df_a, self.comp.df_b)
        with self.assertRaises(AssertionError):
            data = self.fuse._make_resolution_series(
                'age', 'age',
                meta_a='age', meta_b=None,
            )
        with self.assertRaises(AssertionError):
            data = self.fuse._make_resolution_series(
                'age', 'age',
                meta_a=None, meta_b='age',
            )

    def test_resolution_signature_assertions(self):
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.choose_random,
                'values',
                'values',
                remove_na_vals=None
            )
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.choose_random,
                'values',
                'values',
                remove_na_meta=True
            )
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.choose_metadata_max,
                'values',
                'values',
                remove_na_meta=None
            )
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.choose_metadata_max,
                'values',
                'values',
                tie_break=None
            )
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.choose_random,
                'values',
                'values',
                tie_break=cr.choose_random
            )
        with self.assertRaises(AssertionError):
            self.fuse.resolve(
                cr.aggregate,
                'values',
                'values',
                remove_na_vals=True
            )
