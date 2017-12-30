Classification
==============

Classifiers
-----------

Classification is the step in the record linkage process were record pairs are
classified into matches, non-matches and possible matches [Christen2012_].
Classification algorithms can be supervised or unsupervised (rougly speaking:
with or without training data). Many of the algorithms need trainings data to
classify the record pairs. Trainings data is data for which is known whether
it is a match or not.


.. seealso::

    .. [Christen2012] Christen, Peter. 2012. Data matching: concepts and 
        techniques for record linkage, entity resolution, and duplicate 
        detection. Springer Science & Business Media.

.. automodule:: recordlinkage.classifiers
    :members:
    :inherited-members:


Network
-------

The Python Record Linkage Toolkit provides network analysis tools for 
classification of record pairs into matches and distinct pairs. The toolkit 
provides the functionality for one-to-one linking and one-to-many linking. 
It is also possible to detect all connected components which is useful in 
data deduplication. 

.. automodule:: recordlinkage.network
    :members:
    :inherited-members:
