from recordlinkage.base import BaseCompare, BaseIndex
from recordlinkage.compare import Date, Exact, Geographic, Numeric, String
from recordlinkage.index import Block, Full, Random, SortedNeighbourhood


class Index(BaseIndex):
    """Class to make an index of record pairs.

    Parameters
    ----------
    algorithms: list
        A list of index algorithm classes. The classes are based on
        :class:`recordlinkage.base.BaseIndexAlgorithm`

    Example
    -------
    Consider two historical datasets with census data to link. The
    datasets are named ``census_data_1980`` and ``census_data_1990``::

        indexer = recordlinkage.Index()
        indexer.block(left_on='first_name', right_on='givenname')
        indexer.index(census_data_1980, census_data_1990)

    """

    def full(self):
        """Add a 'full' index.

        Shortcut of :class:`recordlinkage.index.Full`::

            from recordlinkage.index import Full

            indexer = recordlinkage.Index()
            indexer.add(Full())

        """
        indexer = Full()
        self.add(indexer)

        return self

    def block(self, *args, **kwargs):
        """Add a block index.

        Shortcut of :class:`recordlinkage.index.Block`::

            from recordlinkage.index import Block

            indexer = recordlinkage.Index()
            indexer.add(Block())

        """
        indexer = Block(*args, **kwargs)
        self.add(indexer)

        return self

    def sortedneighbourhood(self, *args, **kwargs):
        """Add a Sorted Neighbourhood Index.

        Shortcut of :class:`recordlinkage.index.SortedNeighbourhood`::

            from recordlinkage.index import SortedNeighbourhood

            indexer = recordlinkage.Index()
            indexer.add(SortedNeighbourhood())

        """
        indexer = SortedNeighbourhood(*args, **kwargs)
        self.add(indexer)

        return self

    def random(self, *args, **kwargs):
        """Add a random index.

        Shortcut of :class:`recordlinkage.index.Random`::

            from recordlinkage.index import Random

            indexer = recordlinkage.Index()
            indexer.add(Random())

        """
        indexer = Random()
        self.add(indexer)

        return self


class Compare(BaseCompare):
    """Class to compare record pairs with efficiently.

    Class to compare the attributes of candidate record pairs. The
    ``Compare`` class has methods like ``string``, ``exact`` and
    ``numeric`` to initialise the comparing of the records. The
    ``compute`` method is used to start the actual comparing.

    Example
    -------

    Consider two historical datasets with census data to link. The datasets
    are named ``census_data_1980`` and ``census_data_1990``. The MultiIndex
    ``candidate_pairs`` contains the record pairs to compare. The record
    pairs are compared on the first name, last name, sex, date of birth,
    address, place, and income::

        # initialise class
        comp = recordlinkage.Compare()

        # initialise similarity measurement algorithms
        comp.string('first_name', 'name', method='jarowinkler')
        comp.string('lastname', 'lastname', method='jarowinkler')
        comp.exact('dateofbirth', 'dob')
        comp.exact('sex', 'sex')
        comp.string('address', 'address', method='levenshtein')
        comp.exact('place', 'place')
        comp.numeric('income', 'income')

        # the method .compute() returns the DataFrame with the feature vectors.
        comp.compute(candidate_pairs, census_data_1980, census_data_1990)

    Parameters
    ----------
    features : list
        List of compare algorithms.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for comparing of record
        pairs.
        If -1, then the number of jobs is set to the number of cores.
    indexing_type : string, optional (default='label')
        The indexing type. The MultiIndex is used to index the
        DataFrame(s). This can be done with pandas ``.loc`` or with
        ``.iloc``. Use the value 'label' to make use of ``.loc`` and
        'position' to make use of ``.iloc``. The value 'position' is
        only available when the MultiIndex consists of integers. The
        value 'position' is much faster.

    Attributes
    ----------
    features: list
        A list of algorithms to create features.


    """

    def exact(self, *args, **kwargs):
        """Compare attributes of pairs exactly.

        Shortcut of :class:`recordlinkage.compare.Exact`::

            from recordlinkage.compare import Exact

            indexer = recordlinkage.Compare()
            indexer.add(Exact())

        """
        compare = Exact(*args, **kwargs)
        self.add(compare)

        return self

    def string(self, *args, **kwargs):
        """Compare attributes of pairs with string algorithm.

        Shortcut of :class:`recordlinkage.compare.String`::

            from recordlinkage.compare import String

            indexer = recordlinkage.Compare()
            indexer.add(String())

        """
        compare = String(*args, **kwargs)
        self.add(compare)

        return self

    def numeric(self, *args, **kwargs):
        """Compare attributes of pairs with numeric algorithm.

        Shortcut of :class:`recordlinkage.compare.Numeric`::

            from recordlinkage.compare import Numeric

            indexer = recordlinkage.Compare()
            indexer.add(Numeric())

        """
        compare = Numeric(*args, **kwargs)
        self.add(compare)

        return self

    def geo(self, *args, **kwargs):
        """Compare attributes of pairs with geo algorithm.

        Shortcut of :class:`recordlinkage.compare.Geographic`::

            from recordlinkage.compare import Geographic

            indexer = recordlinkage.Compare()
            indexer.add(Geographic())

        """
        compare = Geographic(*args, **kwargs)
        self.add(compare)

        return self

    def date(self, *args, **kwargs):
        """Compare attributes of pairs with date algorithm.

        Shortcut of :class:`recordlinkage.compare.Date`::

            from recordlinkage.compare import Date

            indexer = recordlinkage.Compare()
            indexer.add(Date())

        """
        compare = Date(*args, **kwargs)
        self.add(compare)

        return self
