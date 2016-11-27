from __future__ import absolute_import, division, print_function

import recordlinkage as rl
from recordlinkage.datasets import load_febrl1, load_febrl4


class CompareRecordLinkage(object):

    def setup(self):

        # download data
        self.A, self.B = load_febrl4()

        # make pairs
        c_pairs = rl.Pairs(self.A, self.B)
        pairs = c_pairs.full()

        # different sizes of pairs
        self.pairs_xsmall = pairs[0:5e3]
        self.pairs_small = pairs[0:5e4]
        self.pairs_medium = pairs[0:5e5]
        self.pairs_large = pairs[0:5e6]

    def time_global_xsmall(self):

        c_compare = rl.Compare(self.pairs_xsmall, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_small(self):

        c_compare = rl.Compare(self.pairs_small, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_medium(self):

        c_compare = rl.Compare(self.pairs_medium, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_large(self):

        c_compare = rl.Compare(self.pairs_large, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)


class CompareDeduplication(object):

    def setup(self):

        # download data
        self.A = load_febrl1()

        # make pairs
        c_pairs = rl.Pairs(self.A)
        pairs = c_pairs.full()

        # different sizes of pairs
        self.pairs_xsmall = pairs[0:5e3]
        self.pairs_small = pairs[0:5e4]
        self.pairs_medium = pairs[0:5e5]
        self.pairs_large = pairs[0:5e6]

    def time_global_xsmall(self):

        c_compare = rl.Compare(self.pairs_xsmall, self.A)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_small(self):

        c_compare = rl.Compare(self.pairs_small, self.A)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_medium(self):

        c_compare = rl.Compare(self.pairs_medium, self.A)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)

    def time_global_large(self):

        c_compare = rl.Compare(self.pairs_large, self.A)
        c_compare.string('given_name', 'given_name', method='jaro')
        c_compare.string('surname', 'surname', method='jarowinkler', threshold=0.85)
        c_compare.date('date_of_birth', 'date_of_birth')
        c_compare.exact('suburb', 'suburb')
        c_compare.exact('state', 'state')
        c_compare.string('address_1', 'address_1', method='levenshtein', threshold=0.85)


class CompareAlgorithms(object):

    def setup(self):

        # download data
        self.A, self.B = load_febrl4()

        # make pairs
        c_pairs = rl.Pairs(self.A, self.B)
        self.pairs = c_pairs.full()[0:5e4]

    def time_string_jaro(self):

        c_compare = rl.Compare(self.pairs, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jaro')

    def time_string_jarowinkler(self):

        c_compare = rl.Compare(self.pairs, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='jarowinkler')

    def time_string_levenshtein(self):

        c_compare = rl.Compare(self.pairs, self.A, self.B)
        c_compare.string('given_name', 'given_name', method='levenshtein')

    def time_exact(self):

        c_compare = rl.Compare(self.pairs, self.A, self.B)
        c_compare.exact('state', 'state')
