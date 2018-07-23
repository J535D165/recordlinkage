*****************
3. Classification
*****************

Classifiers
===========

Classification is the step in the record linkage process were record pairs are
classified into matches, non-matches and possible matches [Christen2012_].
Classification algorithms can be supervised or unsupervised (with or without
training data).


.. seealso::

    .. [Christen2012] Christen, Peter. 2012. Data matching: concepts and 
        techniques for record linkage, entity resolution, and duplicate 
        detection. Springer Science & Business Media.

Supervised
----------

.. autoclass:: recordlinkage.LogisticRegressionClassifier 
    :members:
    :inherited-members:

.. autoclass:: recordlinkage.NaiveBayesClassifier
    :members:
    :inherited-members:

.. autoclass:: recordlinkage.SVMClassifier
    :members:
    :inherited-members:

Unsupervised
------------

.. autoclass:: recordlinkage.ECMClassifier
    :members:
    :inherited-members:

.. autoclass:: recordlinkage.KMeansClassifier
    :members:
    :inherited-members:


User-defined algorithms
=======================

Classifiers can make use of the :class:`recordlinkage.base.BaseClassifier` for
algorithms. ScitKit-learn based models may want
:class:`recordlinkage.adapters.SKLearnAdapter` as subclass as well.

.. autoclass:: recordlinkage.base.BaseClassifier

Probabilistic models can use the Fellegi and Sunter base class. This class is
used for the :class:`recordlinkage.ECMClassifier` and the
:class:`recordlinkage.NaiveBayesClassifier`.

.. autoclass:: recordlinkage.classifiers.FellegiSunter

Examples
========

Unsupervised learning with the ECM algorithm. [See example on Github.](https://github.com/J535D165/recordlinkage/examples/unsupervised_learning.py)


Network
=======

The Python Record Linkage Toolkit provides network/graph analysis tools for
classification of record pairs into matches and distinct pairs. The toolkit
provides the functionality for one-to-one linking and one-to-many linking. It
is also possible to detect all connected components which is useful in  data
deduplication.

.. autoclass:: recordlinkage.OneToOneLinking 
.. autoclass:: recordlinkage.OneToManyLinking
.. autoclass:: recordlinkage.ConnectedComponents
