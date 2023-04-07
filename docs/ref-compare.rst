************
2. Comparing
************

A set of informative, discriminating and independent features is important for
a good classification of record pairs into matching and distinct pairs. The
:class:`recordlinkage.Compare` class and its methods can be used to compare records
pairs. Several comparison methods are included such as string similarity
measures, numerical measures and distance measures.


:class:`recordlinkage.Compare` object
=====================================

.. autoclass:: recordlinkage.Compare

    .. automethod:: recordlinkage.Compare.add
    .. automethod:: recordlinkage.Compare.compute

    .. automethod:: recordlinkage.Compare.compare_vectorized
    .. automethod:: recordlinkage.Compare.exact
    .. automethod:: recordlinkage.Compare.string
    .. automethod:: recordlinkage.Compare.numeric
    .. automethod:: recordlinkage.Compare.geo
    .. automethod:: recordlinkage.Compare.date


Algorithms
==========

.. automodule:: recordlinkage.compare
    :members:
    :inherited-members:

User-defined algorithms
=======================

A user-defined algorithm can be defined based on
:class:`recordlinkage.base.BaseCompareFeature`. The :class:`recordlinkage.base.BaseCompareFeature` class is an abstract base
class that is used for compare algorithms. The classes 

- :class:`recordlinkage.compare.Exact`
- :class:`recordlinkage.compare.String`
- :class:`recordlinkage.compare.Numeric`
- :class:`recordlinkage.compare.Date`

are inherited from this abstract base class. You can use BaseCompareFeature to
create a user-defined/custom algorithm. Overwrite the abstract method
:meth:`recordlinkage.base.BaseCompareFeature._compute_vectorized` with the
compare algorithm. A short example is given here:

.. code:: python

    from recordlinkage.base import BaseCompareFeature

    class CustomFeature(BaseCompareFeature):

        def _compute_vectorized(s1, s2):
            # algorithm that compares s1 and s2

            # return a pandas.Series
            return ... 

    feat = CustomFeature()
    feat.compute(pairs, dfA, dfB)

A full description of the :class:`recordlinkage.base.BaseCompareFeature`
class:

.. autoclass:: recordlinkage.base.BaseCompareFeature

    .. automethod:: recordlinkage.base.BaseCompareFeature.compute
    .. automethod:: recordlinkage.base.BaseCompareFeature._compute
    .. automethod:: recordlinkage.base.BaseCompareFeature._compute_vectorized

.. warning:: 

    Do not change the order of the pairs in the MultiIndex.


Examples
========

Example: High level usage
-------------------------


.. code:: python

    import recordlinkage as rl

    comparer = rl.Compare()
    comparer.string('name_a', 'name_b', method='jarowinkler', threshold=0.85, label='name')
    comparer.exact('sex', 'gender', label='gender')
    comparer.date('dob', 'date_of_birth', label='date')
    comparer.string('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7, label='streetname')
    comparer.exact('place', 'placename', label='placename')
    comparer.numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5, label='income')
    comparer.compute(pairs, dfA, dfB)


Example: Low level usage
------------------------


.. code:: python

    import recordlinkage as rl
    from recordlinkage.compare import Exact, String, Numeric, Date

    comparer = rl.Compare([
        String('name_a', 'name_b', method='jarowinkler', threshold=0.85, label='name'),
        Exact('sex', 'gender', label='gender'),
        Date('dob', 'date_of_birth', label='date'),
        String('str_name', 'streetname', method='damerau_levenshtein', threshold=0.7, label='streetname'),
        Exact('place', 'placename', label='placename'),
        Numeric('income', 'income', method='gauss', offset=3, scale=3, missing_value=0.5, label='income'),
    ])
    comparer.compute(pairs, dfA, dfB)



The following examples give a feeling on the extensibility of the toolkit.

Example: User-defined algorithm 1
---------------------------------

The following code defines a custom algorithm to compare zipcodes. The
algorithm returns 1.0 for record pairs that agree on the zipcode and returns
0.0 for records that disagree on the zipcode. If the zipcodes disagree but the
first two numbers are identical, then the algorithm returns 0.5.

.. code:: python

    import recordlinkage as rl
    from recordlinkage.base import BaseCompareFeature

    class CompareZipCodes(BaseCompareFeature):

        def _compute_vectorized(self, s1, s2):
            """Compare zipcodes.

            If the zipcodes in both records are identical, the similarity 
            is 1. If the first two values agree and the last two don't, then 
            the similarity is 0.5. Otherwise, the similarity is 0.
            """
            
            # check if the zipcode are identical (return 1 or 0)
            sim = (s1 == s2).astype(float)
            
            # check the first 2 numbers of the distinct comparisons
            sim[(sim == 0) & (s1.str[0:2] == s2.str[0:2])] = 0.5
            
            return sim

    comparer = rl.Compare()
    comparer.extact('given_name', 'given_name', 'y_name')
    comparer.string('surname', 'surname', 'y_surname')
    comparer.add(CompareZipCodes('postcode', 'postcode', label='y_postcode'))
    comparer.compute(pairs, dfA, dfB)


.. parsed-literal::

    0.0    71229
    0.5     3166
    1.0     2854
    Name: sim_postcode, dtype: int64

.. note:: 
    
    See :class:`recordlinkage.base.BaseCompareFeature` for more
    details on how to subclass.

Example: User-defined algorithm 2
---------------------------------


As you can see, one can pass the labels of the columns as arguments. The
first argument is a column label, or a list of column labels, found in
the first DataFrame (``postcode`` in this example). The second argument
is a column label, or a list of column labels, found in the second
DataFrame (also ``postcode`` in this example). The
``recordlinkage.Compare`` class selects the columns with the given
labels before passing them to the custom algorithm/function. The
``compare`` method in the ``recordlinkage.Compare`` class passes
additional (keyword) arguments to the custom function.

**Warning:** Do not change the order of the pairs in the MultiIndex.


.. code:: python

    import recordlinkage as rl
    from recordlinkage.base import BaseCompareFeature

    class CompareZipCodes(BaseCompareFeature):

        def __init__(self, left_on, right_on, partial_sim_value, *args, **kwargs):
            super(CompareZipCodes, self).__init__(left_on, right_on, *args, **kwargs)

            self.partial_sim_value = partial_sim_value

        def _compute_vectorized(self, s1, s2):
            """Compare zipcodes.

            If the zipcodes in both records are identical, the similarity 
            is 0. If the first two values agree and the last two don't, then 
            the similarity is 0.5. Otherwise, the similarity is 0.
            """
            
            # check if the zipcode are identical (return 1 or 0)
            sim = (s1 == s2).astype(float)
            
            # check the first 2 numbers of the distinct comparisons
            sim[(sim == 0) & (s1.str[0:2] == s2.str[0:2])] = self.partial_sim_value
            
            return sim

    comparer = rl.Compare()
    comparer.extact('given_name', 'given_name', 'y_name')
    comparer.string('surname', 'surname', 'y_surname')
    comparer.add(CompareZipCodes('postcode', 'postcode', 
                                 'partial_sim_value'=0.5, label='y_postcode'))
    comparer.compute(pairs, dfA, dfB)


Example: User-defined algorithm 3
---------------------------------

The Python Record Linkage Toolkit supports the comparison of more than
two columns. This is especially useful in situations with
multi-dimensional data (for example geographical coordinates) and
situations where fields can be swapped.

The FEBRL4 dataset has two columns filled with address information
(``address_1`` and ``address_2``). In a naive approach, one compares
``address_1`` of file A with ``address_1`` of file B and ``address_2``
of file A with ``address_2`` of file B. If the values for ``address_1``
and ``address_2`` are swapped during the record generating process, the
naive approach considers the addresses to be distinct. In a more
advanced approach, ``address_1`` of file A is compared with
``address_1`` and ``address_2`` of file B. Variable ``address_2`` of
file A is compared with ``address_1`` and ``address_2`` of file B. This
is done with the single function given below.

.. code:: python

    import recordlinkage as rl
    from recordlinkage.base import BaseCompareFeature

    class CompareAddress(BaseCompareFeature):

        def _compute_vectorized(self, s1_1, s1_2, s2_1, s2_2):
            """Compare addresses.

            Compare addresses. Compare address_1 of file A with 
            address_1 and address_2 of file B. The same for address_2
            of dataset 1. 
            
            """
            
            return ((s1_1 == s2_1) | (s1_2 == s2_2) | (s1_1 == s2_2) | (s1_2 == s2_1)).astype(float)

    comparer = rl.Compare()

    # naive
    comparer.add(CompareAddress('address_1', 'address_1', label='sim_address_1'))
    comparer.add(CompareAddress('address_2', 'address_2', label='sim_address_2'))

    # better
    comparer.add(CompareAddress(('address_1', 'address_2'), 
                                ('address_1', 'address_2'), 
                                label='sim_address'
    )

    features = comparer.compute(pairs, dfA, dfB)
    features.mean()

The mean of the cross-over comparison is higher.

.. parsed-literal::

    sim_address_1    0.02488
    sim_address_2    0.02025
    sim_address      0.03566
    dtype: float64
