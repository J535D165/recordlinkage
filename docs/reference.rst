Reference
=========

standardise data
----------------

.. automodule:: recordlinkage.standardise
	:members:

.. automodule:: recordlinkage.standardise.cleaning
	:members:

.. automodule:: recordlinkage.standardise.encoding
	:members:


Make record pairs
-----------------

.. automodule:: recordlinkage.indexing

.. autoclass:: Pairs
	:members:

Comparing record pairs
----------------------

.. automodule:: recordlinkage.comparing

.. autoclass:: Compare
	:members:

Classification algorithms 
-------------------------

.. automodule:: recordlinkage.classifier
	:members:

.. autoclass:: Classifier
	:members:

.. autoclass:: KMeansClassifier
	:members:

.. autoclass:: LogisticRegressionClassifier
	:members:

Evaluation 
----------
Evaluation of classifications plays an important role in record linkage. 

.. automodule:: recordlinkage.measures
	:members:

Datasets
--------

This package contains some example datasets that can be used for experiments. For example a dataset of comparison vectors and their labels for a cancer research. It is also possible to generate fake datasets. 

.. autofunction:: recordlinkage.datasets.krebsregister_cmp_data
