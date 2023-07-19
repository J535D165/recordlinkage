import recordlinkage as rl
from recordlinkage.datasets import load_febrl1
from recordlinkage.datasets import load_febrl4


class PairsRecordLinkage:
    timeout = 30 * 60

    def setup(self):
        # download data
        self.A, self.B = load_febrl4()

    def time_full_index(self):
        # setup class
        c_pairs = rl.FullIndex()

        # Make pairs
        c_pairs.index(self.A, self.B)

    def time_block_index(self):
        # setup class
        c_pairs = rl.BlockIndex("given_name")

        # Make pairs
        c_pairs.index(self.A, self.B)

    def time_sni_index(self):
        # setup class
        c_pairs = rl.SortedNeighbourhoodIndex(on="given_name", w=5)

        # Make pairs
        c_pairs.index(self.A, self.B)

    def time_random_index(self):
        # setup class
        c_pairs = rl.RandomIndex(2500)

        # Make pairs
        c_pairs.index(self.A, self.B)


class PairsDeduplication:
    timeout = 30 * 60

    def setup(self):
        # download data
        self.A = load_febrl1()

    def time_full_index(self):
        # setup class
        c_pairs = rl.FullIndex()

        # Make pairs
        c_pairs.index(self.A)

    def time_block_index(self):
        # setup class
        c_pairs = rl.BlockIndex("given_name")

        # Make pairs
        c_pairs.index(self.A)

    def time_sni_index(self):
        # setup class
        c_pairs = rl.SortedNeighbourhoodIndex(on="given_name", w=5)

        # Make pairs
        c_pairs.index(self.A)

    def time_random_index(self):
        # setup class
        c_pairs = rl.RandomIndex(2500)

        # Make pairs
        c_pairs.index(self.A)
