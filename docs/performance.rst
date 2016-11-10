***********
Performance
***********

Performance plays an important role in record linkage. Smart indexation techniques are developed in the past and implemented in this package. The are much more options to increase the performance of your record linkage implementation. 

Tips and tricks
===============

Some of the options:

Blocking on multiple columns
----------------------------

Blocking is an effective way to increase performance. If the performance of your implementation is poor, increase the speed by blocking on multiple blocking keys. Use ``index.block(['col1', 'col2'], ['col1', 'col2'])``. Oke, you might exclude more links accidently. But you can repeat the process with a different combination of blocking keys. For example, ``index.block(['col1', 'col3'], ['col1', 'col3'])``. Merge in the end the links of the two passes. 

.. code-block:: python

    >>> pcl = recordlinkage.Pairs(dfA, dfB)
    >>> candidate_pairs = pcl.block(left_on=['first_name', 'surname'], right_on=['name', 'surname'])

Compare in a batch
------------------

The structure of the Python Record Linkage Toolkit has a drawback. The indexation step and comparing step are separated from each other. This is not good for the performance, but uses less memory and is better understandable. If you make a lot of comparisons, the performance can be bad. In this case use ``Compare(..., batch=True)``. 

.. code-block:: python

    >>> # This example is almost 3 times faster than the traditional one.
    >>> comp = recordlinkage.Compare(..., batch=True)
    >>> comp.exact('first_name', 'name')
    >>> comp.exact('surname', 'surname')
    >>> comp.exact('date_of_birth', 'dob')
    >>> comp.run()

See http://recordlinkage.readthedocs.io/en/latest/reference.html#recordlinkage.comparing.Compare.run.

Split the indexation step
-------------------------

In some cases, it helps to split the files before indexation. The Python Record Linkage Toolkit has a built-in tool for this. Read http://recordlinkage.readthedocs.io/en/latest/notebooks/advanced_indexing.html#Indexing-with-large-files for the example. 

Split the indexation step and run in parallel
---------------------------------------------

Split the indexation step like discussed in http://recordlinkage.readthedocs.io/en/latest/notebooks/advanced_indexing.html#Indexing-with-large-files. Compare each chunk in parallel. 

Compare only discriminating variables
-------------------------------------

Not all variables are worth comparing in a record linkage. Some of them are not discriminating the links of the non-links. These variables can be excluded. Only discriminating and informative should be included. 

Prevent string comparisons
--------------------------

If the number of candidate links is larger than the number of records in both datasets together, than think about phonetic encoding of string variables instead of string comparison. String comparisons are expensive, but may lead to better results. 

String comparison
-----------------

Comparing strings is expensive. The Python Record Linkage Toolkit uses ``jellyfish`` for string comparison. The package has two implementations, a C and a Python implementation. Ensure yourself of having the C-version installed.  There is a large difference in performance between the string comparison methods. The Jaro and Jaro-Winkler methods are faster than the Levenshtein distance and much faster than the Damerau-Levenshtein distance. 

Do you know more tricks? Let us know!