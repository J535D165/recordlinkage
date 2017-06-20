Indexing
========

**Migrating from recordlinkage<=0.8.2 to recordlinkage>=0.9?** `Click here 
<#migrating>`_

The indexing module is used to make pairs of records. These pairs are called
candidate links or candidate matches. There are several indexing algorithms
available such as blocking and sorted neighborhood indexing. See the following
references for background information about indexation.

.. [christen2012] Christen, P. (2012). Data matching: concepts and
        techniques for record linkage, entity resolution, and duplicate
        detection. Springer Science & Business Media.
.. [christen2008] Christen, P. (2008). Febrl - A Freely Available Record
        Linkage System with a Graphical User Interface.


.. automodule:: recordlinkage.indexing

.. autoclass:: FullIndex
	:members:
	:inherited-members:

.. autoclass:: BlockIndex
	:members:
	:inherited-members:

.. autoclass:: SortedNeighbourhoodIndex
	:members:
	:inherited-members:

.. autoclass:: RandomIndex
	:members:
	:inherited-members:


Migrating
---------

Version 0.9 of the Python Record Linkage Toolkit uses a new indexing API. The
new indexing API uses a different syntax. With the new API, each algorithm
has it's own class. See the following example to migrate a blocking index:

Old (linking): 

.. code:: python

	cl = recordlinkage.Pairs(df_a, df_b)
	cl.block('given_name')


New (linking):

.. code:: python

	cl = recordlinkage.BlockIndex('given_name')
	cl.index(df_a, df_b)

Old (deduplication): 

.. code:: python

	cl = recordlinkage.Pairs(df_a)
	cl.block('given_name')


New (deduplication):

.. code:: python

	cl = recordlinkage.BlockIndex('given_name')
	cl.index(df_a)

