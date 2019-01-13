Python Record Linkage Toolkit contrib
=====================================

Any code in this directory is not officially supported, and may change
or be removed at any time without notice. It is meant to contain
algorithms and contributions that eventually should get merged into core
of the toolkit, but whose interfaces may still change, or which require
some testing to see whether they can find broader acceptance.

Examples
--------

.. code:: python

   from recordlinkage.contrib.index import NeighbourhoodBlock

   # or 

   from recordlinkage.contrib.compare.random import RandomContinuous

Development
-----------

Please add single algorithms directly to ``recordlinkage.contrib.index``
or ``recordlinkage.contrib.compare``, but collections of algorithms as a
submodule.