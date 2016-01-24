
Advanced indexing
=================

The ``recordlinkage`` module contains several build in indexing
alogorithms like *full indexing*, *blocking* and *sorted neighbourhood
indexing*. Sometimes, these indexing methods are not sufficient. With
the ``recordlinkage`` module, it is very easy to implement your own
indexing algorithms. In this example, we will show how you can implement
your own indexing algorithm.

Import the module and two sample datasets. Also ``numpy`` and ``pandas``
is used for this example.

.. code:: python

    import numpy
    import pandas
    
    import recordlinkage
    from recordlinkage.datasets import load_censusA, load_censusB
    
    dfA = load_censusA()
    dfB = load_censusB()

Use the class ``Pairs`` to make the record pairs.

.. code:: python

    pcl = recordlinkage.Pairs(dfA, dfB)

So far, nothing changed. To make a custom indexing algorithm, we have to
make a function that does the work for us. In the following example, a
random indexing algorithm is made. The algorithm makes record pairs
where each record in the record pair in sampled randomly of dataframe
``dfA`` or ``dfB``.

.. code:: python

    def randomindex(A, B, N_pairs):
    
        random_index_A = numpy.random.choice(A.index.values, N_pairs)
        random_index_B = numpy.random.choice(B.index.values, N_pairs)
    
        index = pandas.MultiIndex.from_tuples(zip(random_index_A, random_index_B), names=[A.index.name, B.index.name])
    
        return index.drop_duplicates()

This function takes the two dataframes ``A`` and ``B`` as input
arguments. The argument ``N_pairs`` defines the number of maximum number
of record pairs returned. The lines

::

    random_index_A = np.random.choice(A.index.values, N_pairs)
    random_index_B = np.random.choice(B.index.values, N_pairs)

are used to sample random indices from both DataFrames. Note that the
DataFrames ``A`` and ``B`` are ``pandas.Dataframes``, so we can make
advantage of that. In the next line,

::

    index = pd.MultiIndex.from_tuples(zip(random_index_A, random_index_B), names=[A.index.name, B.index.name])

we make a ``pandas.MultiIndex``. See
http://pandas.pydata.org/pandas-docs/stable/advanced.html for more
details about making and using a MultiIndex. The function returns

::

    index.drop_duplicates()

The duplicates are dropped (so we may not return exactly ``N_pairs``).
This is because the MultiIndex has to be unique (we are not going to
compare the same record pairs more than once in the comparison step)

With the function ``randomindex``, it is possible to make record pairs
directly. Just by calling ``randomindex(dfA, dfB, 1000)``. In that case,
it is not possible to use the other build methods in the ``Pairs``
class. Therefore, we can call the following method:

.. code:: python

    random_record_pairs = pcl.index(randomindex, 1000)

The number of random record pairs is equal or slightly less than 1000.

.. code:: python

    print(len(random_record_pairs))


.. parsed-literal::

    1000


Indexing with large files
-------------------------

Sometimes, the input files are very large. In that case, it can be hard
to make an index without running out of memory in the indexing step or
in the comparing step. ``recordlinkage`` has a method to deal with large
files. It is fast, although is not primary developed to be fast. SQL
databases may outperform this method. It is especially developed for the
useability. The idea was to spllit the input files into small blocks.
For each block the record pairs are computed. Then iterate over the
blocks. Consider full indexing:

.. code:: python

    for index_block in pcl.iterfull(500,500):
        
        # Index returned
        print(type(index_block))
    
        # Length of index block
        print(len(index_block))
        
        # Your analysis here


.. parsed-literal::

    <class 'pandas.core.index.MultiIndex'>
    250000
    <class 'pandas.core.index.MultiIndex'>
    250000
    <class 'pandas.core.index.MultiIndex'>
    250000
    <class 'pandas.core.index.MultiIndex'>
    250000


The values 500 and 500 are the number of records used from ``dfA`` and
``dfB`` respectivily. So if both files contain 1000 records, there are
four blocks.

Each implemented indexing algorithm has iterative variant. If the
function need additional parameters, they can be passed after the block
size parameters. For example:

::

    pcl.iterblock(500,500, 'first_name')
    pcl.itersortedneighbourhood(250, 500, 'first_name', 3)

It is also possible to use iterative indexing for your own functions! It
works just in the same way:

.. code:: python

    for index_block in pcl.iterindex(randomindex, 500, 500, 1000):
        
        # Index returned
        print(type(index_block))
    
        # Length of index block
        print(len(index_block))
        
        # Your analysis here


.. parsed-literal::

    <class 'pandas.core.index.MultiIndex'>
    997
    <class 'pandas.core.index.MultiIndex'>
    996
    <class 'pandas.core.index.MultiIndex'>
    997
    <class 'pandas.core.index.MultiIndex'>
    999

