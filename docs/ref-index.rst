***********
1. Indexing
***********

The indexing module is used to make pairs of records. These pairs are called
candidate links or candidate matches. There are several indexing algorithms
available such as blocking and sorted neighborhood indexing. See
[christen2012]_ and [christen2008]_ for background information about
indexation.

.. [christen2012] Christen, P. (2012). Data matching: concepts and
        techniques for record linkage, entity resolution, and duplicate
        detection. Springer Science & Business Media.
.. [christen2008] Christen, P. (2008). Febrl - A Freely Available Record
        Linkage System with a Graphical User Interface.

The indexing module can be used for both linking and duplicate detection. In
case of duplicate detection, only pairs in the upper triangular part of the
matrix are returned. This means that the first record in each record pair is
the largest identifier. For example, `("A2", "A1")`, `(5, 2)` and  `("acb",
"abc")`. The following image shows the record pairs for a complete set of
record pairs.

.. figure:: /images/indexing_basic.png
    :width: 100%

:class:`recordlinkage.Index` object
===================================

.. autoclass:: recordlinkage.Index

    .. automethod:: recordlinkage.Index.add
    .. automethod:: recordlinkage.Index.index
    .. automethod:: recordlinkage.Index.full
    .. automethod:: recordlinkage.Index.block
    .. automethod:: recordlinkage.Index.sortedneighbourhood
    .. automethod:: recordlinkage.Index.random



Algorithms
==========

The Python Record Linkage Toolkit contains basic and advanced indexing (or
blocking) algorithms to make record pairs. The algorithms are Python classes.
Popular algorithms in the toolkit are:

- :class:`recordlinkage.index.Full`,
- :class:`recordlinkage.index.Block`,
- :class:`recordlinkage.index.SortedNeighbourhood`

The algorithms are available in the submodule `recordlinkage.index`. Import
the algorithms in the following way (use blocking algorithm as example):

.. code:: python

    from recordlinkage.index import Block

The full reference for the indexing algorithms in the toolkit is given below.

.. automodule:: recordlinkage.index
    :members:
    :inherited-members:

User-defined algorithms
=======================

A user-defined algorithm can be defined based on
:class:`recordlinkage.base.BaseIndexAlgorithm`. The :class:`recordlinkage.base.BaseIndexAlgorithm` class is an abstract base
class that is used for indexing algorithms. The classes

- :class:`recordlinkage.index.Full`
- :class:`recordlinkage.index.Block`
- :class:`recordlinkage.index.SortedNeighbourhood`
- :class:`recordlinkage.index.Random`

are inherited from this abstract base class. You can use BaseIndexAlgorithm to
create a user-defined/custom algorithm.

To create a custom algorithm, subclass the
:class:`recordlinkage.base.BaseIndexAlgorithm`. In the subclass, overwrite the
:meth:`recordlinkage.base.BaseIndexAlgorithm._link_index` method in case of
linking two datasets. This method accepts two (tuples of)
:class:`pandas.Series` objects as arguments. Based on these Series objects,
you create record pairs. The record pairs need to be returned in a 2-level
:class:`pandas.MultiIndex` object. The :attr:`pandas.MultiIndex.names` are the
name of index of DataFrame A and name of the index of DataFrame B
respectively. Overwrite the
:meth:`recordlinkage.base.BaseIndexAlgorithm._dedup_index` method in case of
finding link within a single dataset (deduplication). This method accepts a
single (tuples of) :class:`pandas.Series` objects as arguments.

The algorithm for linking data frames can be used for finding duplicates as
well. In this situation, DataFrame B is a copy of DataFrame A. The Pairs class
removes pairs like (record_i, record_i) and one of the following (record_i,
record_j) (record_j, record_i) under the hood. As result of this, only unique
combinations are returned. If you do have a specific algorithm for finding
duplicates, then you can overwrite the _dedup_index method. This method
accepts only one argument (DataFrame A) and the internal base class does not
look for combinations like explained above.

.. autoclass:: recordlinkage.base.BaseIndexAlgorithm
    :members:
    :private-members:

Examples
========

.. code:: python

    import recordlinkage as rl
    from recordlinkage.datasets import load_febrl4
    from recordlinkage.index import Block

    df_a, df_b = load_febrl4()

    indexer = rl.Index()
    indexer.add(Block('given_name', 'given_name'))
    indexer.add(Block('surname', 'surname'))
    indexer.index(df_a, df_b)

Equivalent code:

.. code:: python

    import recordlinkage as rl
    from recordlinkage.datasets import load_febrl4

    df_a, df_b = load_febrl4()

    indexer = rl.Index()
    indexer.block('given_name', 'given_name')
    indexer.block('surname', 'surname')
    index.index(df_a, df_b)

This example shows how to implement a custom indexing algorithm. The algorithm
returns all record pairs of which the given names starts with the letter ‘W’.

.. code:: python

    import recordlinkage
    from recordlinkage.datasets import load_febrl4

    df_a, df_b = load_febrl4()

    from recordlinkage.base import BaseIndexAlgorithm

    class FirstLetterWIndex(BaseIndexAlgorithm):
        """Custom class for indexing"""

        def _link_index(self, df_a, df_b):
            """Make pairs with given names starting with the letter 'w'."""

            # Select records with names starting with a w.
            name_a_w = df_a[df_a['given_name'].str.startswith('w') == True]
            name_b_w = df_b[df_b['given_name'].str.startswith('w') == True]

            # Make a product of the two numpy arrays
            return pandas.MultiIndex.from_product(
                [name_a_w.index.values, name_b_w.index.values],
                names=[df_a.index.name, df_b.index.name]
            )

    indexer = FirstLetterWIndex()
    candidate_pairs = indexer.index(df_a, df_b)

    print ('Returns a', type(candidate_pairs).__name__)
    print ('Number of candidate record pairs starting with the letter w:', len(candidate_pairs))

The custom index class below does not restrict the first letter to ‘w’, but the first letter is an argument (named `letter`). This letter can is initialized during the setup of the class.

.. code:: python

    class FirstLetterIndex(BaseIndexAlgorithm):
        """Custom class for indexing"""

        def __init__(self, letter):
            super(FirstLetterIndex, self).__init__()

            # the letter to save
            self.letter = letter

        def _link_index(self, df_a, df_b):
            """Make record pairs that agree on the first letter of the given name."""

            # Select records with names starting with a 'letter'.
            a_startswith_w = df_a[df_a['given_name'].str.startswith(self.letter) == True]
            b_startswith_w = df_b[df_b['given_name'].str.startswith(self.letter) == True]

            # Make a product of the two numpy arrays
            return pandas.MultiIndex.from_product(
                [a_startswith_w.index.values, b_startswith_w.index.values],
                names=[df_a.index.name, df_b.index.name]
            )
