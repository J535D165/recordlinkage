*************************
ScitKit-Learn classifiers
*************************

The Python Record Linkage Toolkit ships with multiple classifiers. These
classifiers make it easy to distinguish matches and non-matches. The built-in
classifiers may not fit your needs. This section describes how to use custom
classifiers.

SciKit-Learn base class
=======================

The Python Record Linkage Toolkit has a special class for ScitKit-Learn
kernels. The class :class:`recordlinkage.adapters.SKLearnAdapter` makes it easy
to use an ScitKit-Learn classifier.

.. code:: python
    
    # import ScitKit-Learn classifier
    from sklearn.ensemble import RandomForestClassifier

    # import BaseClassifier from recordlinkage.base
    from recordlinkage.base import BaseClassifier
    from recordlinkage.adapters import SKLearnClassifier

    class RandomForest(SKLearnClassifier, BaseClassifier):

        def __init__(*args, **kwargs):
            super(self, RandomForest).__init__()

            # set the kernel
            kernel = RandomForestClassifier(*args, **kwargs)


.. code:: python

    from recordlinkage.datasets import binary_vectors

    # make a sample dataset
    features, links = binary_vectors(10000, 2000, return_links=True)

    # initialise the random forest
    cl = RandomForest(n_estimators=20)
    cl.fit(features, links)

    # predict the matches
    cl.predict(...)
    

