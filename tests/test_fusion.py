import unittest
import datetime
import warnings
import functools
import itertools as it
from pprint import pformat
from typing import Callable
from random import randrange
from datetime import timedelta
from parameterized import parameterized, param

import recordlinkage
from recordlinkage import rl_logging

import pandas
import numpy as np
from numpy import nan, arange

# rl_logging.set_verbosity(rl_logging.INFO)

# TODO: What's the deal with code coverage?


def validate_job(job: dict):
    def if_not_none(key: str, f: Callable[..., bool], *args, **kwargs):
        return f(job[key], *args, **kwargs) if job[key] is not None else True

    def warn_on_fail(cond: bool, msg: str):
        if cond is False:
            warnings.warn(msg)
        return cond

    handler = job['handler'].__name__

    checks = [
        # The job
        warn_on_fail(isinstance(job, dict), 'job not dict'),
        # CR Function
        warn_on_fail('fun' in job.keys(), 'no resolution function'),
        warn_on_fail(callable(job['fun']), ' resolution function not callable'),
        # Handling function
        warn_on_fail('handler' in job.keys(), 'no handling method'),
        warn_on_fail(callable(job['handler']), 'handling method not callable'),
        # Data columns
        warn_on_fail(job['values_a'] is not None if handler is not '_do_keep' else True,
                     'values_a is None when not keeping original columns'),
        warn_on_fail(job['values_b'] is not None if handler is not '_do_keep' else True,
                     'values_b is None when not keeping original columns'),
        warn_on_fail(job['values_a'] is not None or job['values_b'] is not None if handler is not '_do_keep' else True,
                     'both values_a and values_b are None'),
        warn_on_fail(if_not_none('values_a', isinstance, (str, list)), 'bad values_a type'),
        warn_on_fail(if_not_none('values_b', isinstance, (str, list)), 'bad vlaues_b type'),
        # Metadata columns
        warn_on_fail(if_not_none('meta_a', isinstance, (str, list)), 'bad meta_a type'),
        warn_on_fail(if_not_none('meta_b', isinstance, (str, list)), 'bad meta_b type'),
        # Transformation functions
        warn_on_fail(if_not_none('transform_vals', callable), 'transform_vals not callable'),
        warn_on_fail(if_not_none('transform_meta', callable), 'transform_meta not callable'),
        # Args and Kwargs
        warn_on_fail(if_not_none('params', isinstance, tuple), 'bad params type'),
        warn_on_fail(if_not_none('kwargs', isinstance, dict), 'bad kwargs type'),
        # Name
        warn_on_fail(if_not_none('name', isinstance, str), 'bad name type'),
        # Description
        warn_on_fail(if_not_none('description', isinstance, str), 'bad description type')
    ]
    result = functools.reduce(lambda a, b: a and b, checks)
    warn_on_fail(result, 'job failed:\n' + pformat(job))
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
TIE_BREAK_SRATEGIES = [
    {'method': 'cry_with_the_wolves', 'params': [], 'kws': {}},
    {'method': 'keep_up_to_date', 'params': ['date', 'date'], 'kws': {}}
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
    [param(case[0]['method'], case[1], case[2], *case[0]['params'], **case[0]['kws'])
     for case in it.product(GENERAL_STATEGIES, STR_COLNAMES, STR_COLNAMES)],
    [param(case[0]['method'], case[1], case[2], *case[0]['params'], tie_break=case[3], **case[0]['kws'])
     for case in it.product(TIE_BREAK_SRATEGIES, STR_COLNAMES, STR_COLNAMES, TIE_BREAK_OPTIONS)],
    [param(case[0]['method'], case[1], case[2], case[3], *case[0]['params'], **case[0]['kws'])
     for case in it.product(NUMERIC_STRATEGIES, NUM_COLNAMES, NUM_COLNAMES, NUMERIC_OPTIONS)],
)


class TestFuseLinks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        N_A = 100
        N_B = 100

        cls.A = pandas.DataFrame({
            'age': np.random.choice(AGES, N_A),
            'given_name': np.random.choice(FIRST_NAMES, N_A),
            'last_name': np.random.choice(LAST_NAMES, N_A),
            'street': np.random.choice(STREET, N_A),
            'income': np.random.choice(INCOMES, N_A),
            'date': np.random.choice(DATES, N_A)
        })

        cls.B = pandas.DataFrame({
            'age': np.random.choice(AGES, N_B),
            'given_name': np.random.choice(FIRST_NAMES, N_B),
            'last_name': np.random.choice(LAST_NAMES, N_B),
            'street': np.random.choice(STREET, N_B),
            'income': np.random.choice(INCOMES, N_B),
            'date': np.random.choice(DATES, N_B)
        })

        cls.A.index.name = 'index_df1'
        cls.B.index.name = 'index_df2'

        cls.index_AB = pandas.MultiIndex.from_arrays(
            [arange(len(cls.A)), arange(len(cls.B))],
            names=[cls.A.index.name, cls.B.index.name])

    @parameterized.expand(RESOLUTION_CASES)
    def test_resolution_result(self, method_to_call, *args, **kwargs):
        """conflict resolution result is a pandas series"""
        comp = recordlinkage.Compare(self.index_AB, self.A, self.B)
        fuse = recordlinkage.FuseLinks()
        getattr(fuse, method_to_call)(*args, **kwargs)
        # Validate the job metadata
        self.assertTrue(validate_job(fuse.resolution_queue[0]))
        # Check job runs and produces dataframe.
        result = fuse.fuse(comp.vectors, self.A, self.B)
        self.assertIsInstance(result, pandas.DataFrame, 'result not a dataframe')