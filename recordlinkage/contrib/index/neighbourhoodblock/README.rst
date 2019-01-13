Neighbourhood blocking
======================

Example
-------

In the following example, the record pairs are made for two historical
datasets with census data. The datasets are named ``census_data_1980``
and ``census_data_1990``. The index includes record pairs with matches
in (at least) any 3 out of the 5 nominated fields. Proximity matching is
allowed in the first two fields, and up to one wildcard match of a
missing value is also allowed.

.. code:: python

   from recordlinkage.contrib.index import NeighbourhoodBlock

   keys = ['first_name', 'surname', 'date_of_birth', 'address', 'ssid']
   windows = [9, 3, 1, 1, 1]

   indexer = NeighbourhoodBlock(
       keys, windows=windows, max_nulls=1, max_non_matches=2)
   indexer.index(census_data_1980, census_data_1990)

Authors
-------

-  Daniel Elias