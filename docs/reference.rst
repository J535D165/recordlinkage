API Reference
=============

This page contains the class, method and functions documentation of the ``recordlinkage`` package. 

Standardising
-------------

Clean and standardise your data before you start linking records. Remove special tokens or phonetically encode names. Install ``jellyfish`` to use phonetic encoding of names. 

.. automodule:: recordlinkage.standardise
	:members:

.. automodule:: recordlinkage.standardise.cleaning
	:members:

.. automodule:: recordlinkage.standardise.encoding
	:members:


Indexing
--------

This class can be used to make pairs of records, also called candidate record pairs. There are several smart indexing functions available. 

.. automodule:: recordlinkage.indexing

.. autoclass:: Pairs
	:members:

Comparing
---------

The ``Compare`` class and its methods can be used to compare records pairs. Several comparison methods are included such as string similarity measures, numerical measures and distance measures. The string similarity algorithms are available when ``jellyfish`` is installed (pip install jellyfish). 

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
Evaluation of classifications plays an important role in record linkage. Express your classification quality in terms accuracy, recall and F-score based on ``true positives``, ``false positives``, ``true negatives`` and ``false negatives``. 

.. automodule:: recordlinkage.measures
	:members:

Datasets
--------

This package contains some example datasets that can be used for experiments. For example a dataset of comparison vectors and their labels for a cancer research. It is also possible to generate fake datasets. 

.. autofunction:: recordlinkage.datasets.krebsregister_cmp_data
