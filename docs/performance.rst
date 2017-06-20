
Performance
===========

Performance plays an important role in record linkage. Record linkage problems
scale quadratically with the size of the file(s). The number of record pairs
can be enormous and so are the number of comparisons. Over the years, smart
techniques are developed to reduce the number of record pairs. The *Python
Record Linkage Toolkit* offers several effective techniques for making record
pairs. Nevertheless, the Python Record Linkage Toolkit is **NOT** developed
with speed in mind. **Understandability**, **usability** and **extensibility**
are the main focus points in the development. This makes the toolkit very
useful for linking small or medium sized files.

Okeee... There is not only bad news. The performance of many algorithms
is good. Nevertheless, there are plenty of options to increase the
performance of your record linkage implementation. Some methods are
discussed in this topic.

Do you know more tricks? Let us know!

Indexation
----------

Block on multiple columns
~~~~~~~~~~~~~~~~~~~~~~~~~

Blocking is an effective way to increase the performance of your record
linkage. If the performance of your implementation is still poor, decrease the
number of pairs by blocking on multiple variables. This implies that the
record pair is agrees on two or more variables. In the following example, the
record pairs agree on the given name **and** surname.

.. code:: python

    p = recordlinkage.BlockIndex(left_on=['first_name', 'surname'], 
                                 right_on=['name', 'surname'])
    pairs = p.index(dfA, dfB)

You might exclude more links then desired. This can be solved by
repeating the process with a different combinations of blocking
variables. In the end, merge the links of the two, so called, index
passes.

.. code:: python

    p_surname = recordlinkage.BlockIndex(left_on=['first_name', 'surname'], 
                                         right_on=['name', 'surname'])
    p_age = recordlinkage.BlockIndex(left_on=['first_name', 'age'], 
                                     right_on=['name', 'age'])

    pairs_surname = p_surname.index(dfA, dfB)
    pairs_age = p_age.index(dfA, dfB)

    # make a union of the candidate links of both classes
    pairs_surname.union(pairs_age)

.. note:: Sorted Neighbourhood indexing supports, besides the sorted
        neighbourhood, additional blocking on variables. 

Make record pairs
~~~~~~~~~~~~~~~~~

The structure of the Python Record Linkage Toolkit has a drawback for the
performance. In the indexation step (the step in which record pairs are
selected), only the index of both records is stored. The entire records
are not stored. This results in less memory usage. The drawback is that the
records need to be queried from the data. 


Comparing
---------

Compare only discriminating variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not all variables may be worth comparing in a record linkage. Some variables
do not discriminate the links of the non-links or do have only minor effects.
These variables can be excluded. Only discriminating and informative should be
included.

Prevent string comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~

String similarity measures and phonetic encodings are computationally
expensive. Phonetic encoding takes place on the original data, while string
simililatiry measures are applied on the record pairs. After phonetic encoding
of the string variables, exact comparing can be used instead of computing the
string similarity of all record pairs. If the number of candidate pairs is
much larger than the number of records in both datasets together, then
consider using phonetic encoding of string variables instead of string
comparison.

String comparing
~~~~~~~~~~~~~~~~

Comparing strings is computationally expensive. The Python Record Linkage
Toolkit uses the package ``jellyfish`` for string comparisons. The package has
two implementations, a C and a Python implementation. Ensure yourself of
having the C-version installed (``import jellyfish.cjellyfish`` should not
raise an exception).

There can be a large difference in the performance of different string
comparison algorithms. The Jaro and Jaro-Winkler methods are faster than the
Levenshtein distance and much faster than the Damerau-Levenshtein distance.

Indexing with large files
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, the input files are very large. In that case, it can be hard
to make an index without running out of memory in the indexing step or
in the comparing step. ``recordlinkage`` has a method to deal with large
files. It is fast, although is not primary developed to be fast. SQL
databases may outperform this method. It is especially developed for the
useability. The idea was to spllit the input files into small blocks.
For each block the record pairs are computed. Then iterate over the
blocks. Consider full indexing:

.. code:: python

    import recordlinkage
    import numpy

    cl = recordlinkage.FullIndex()
    
    for dfB_subset in numpy.split(dfB):
        
        # a subset of record pairs
        pairs_subset = cl.index(dfA, dfB_subset)
        
        # Your analysis on pairs_subset here


