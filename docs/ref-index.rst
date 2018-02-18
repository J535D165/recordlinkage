Indexing
========

The indexing module is used to make pairs of records. These pairs are called
candidate links or candidate matches. There are several indexing algorithms
available such as blocking and sorted neighborhood indexing. See the following
references for background information about indexation.

.. [christen2012] Christen, P. (2012). Data matching: concepts and
        techniques for record linkage, entity resolution, and duplicate
        detection. Springer Science & Business Media.
.. [christen2008] Christen, P. (2008). Febrl - A Freely Available Record
        Linkage System with a Graphical User Interface.


.. autoclass:: recordlinkage.Index
	:inherited-members: add, index, full, block, sortneighbourhood, random


Algorithms
----------

.. automodule:: recordlinkage.index
	:members:
	:inherited-members:

