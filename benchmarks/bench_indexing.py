from __future__ import absolute_import, division, print_function

import recordlinkage as rl
from recordlinkage.datasets import load_febrl1, load_febrl4


class PairsRecordLinkage(object):

    def setup(self):

        # download data
        self.A, self.B = load_febrl4()

    def time_full_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A, self.B)

        # Make pairs
        c_pairs.full()

    def time_block_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A, self.B)

        # Make pairs
        c_pairs.block('given_name')

    def time_sni_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A, self.B)

        # Make pairs
        c_pairs.sortedneighbourhood('given_name', 5)

    def time_random_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A, self.B)

        # Make pairs
        c_pairs.random(2500)


class PairsDeduplication(object):

    def setup(self):

        # download data
        self.A = load_febrl1()

    def time_full_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A)

        # Make pairs
        c_pairs.full()

    def time_block_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A)

        # Make pairs
        c_pairs.block('given_name')

    def time_sni_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A)

        # Make pairs
        c_pairs.sortedneighbourhood('given_name', 5)

    def time_random_index(self):

        # setup class
        c_pairs = rl.Pairs(self.A)

        # Make pairs
        c_pairs.random(2500)
