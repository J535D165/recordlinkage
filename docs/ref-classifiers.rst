*****************
3. Classification
*****************

Classifiers
===========

Classification is the step in the record linkage process were record pairs are
classified into matches, non-matches and possible matches [Christen2012]_.
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


Adapters
========

Adapters can be used to wrap a machine learning models from external packages
like ScitKit-learn and Keras. For example, this makes it possible to classify
record pairs with an neural network developed in Keras.

.. autoclass:: recordlinkage.adapters.SKLearnAdapter


.. code:: python

    # import ScitKit-Learn classifier
    from sklearn.ensemble import RandomForestClassifier

    # import BaseClassifier from recordlinkage.base
    from recordlinkage.base import BaseClassifier
    from recordlinkage.adapters import SKLearnClassifier
    from recordlinkage.datasets import binary_vectors

    class RandomForest(SKLearnClassifier, BaseClassifier):

        def __init__(*args, **kwargs):
            super(self, RandomForest).__init__()

            # set the kernel
            kernel = RandomForestClassifier(*args, **kwargs)


    # make a sample dataset
    features, links = binary_vectors(10000, 2000, return_links=True)

    # initialise the random forest
    cl = RandomForest(n_estimators=20)
    cl.fit(features, links)

    # predict the matches
    cl.predict(...)


.. autoclass:: recordlinkage.adapters.KerasAdapter

Example of a Keras model used for classification.

.. code:: python

    from tensorflow.keras import layers
    from recordlinkage.base import BaseClassifier
    from recordlinkage.adapters import KerasAdapter

    class NNClassifier(KerasAdapter, BaseClassifier):
        """Neural network classifier."""
        def __init__(self):
            super(NNClassifier, self).__init__()

            model = tf.keras.Sequential()
            model.add(layers.Dense(16, input_dim=8, activation='relu'))
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=tf.train.AdamOptimizer(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            self.kernel = model

    # initialise the model
    cl = NNClassifier()
    # fit the model to the data
    cl.fit(X_train, links_true)
    # predict the class of the data
    cl.predict(X_pred)


User-defined algorithms
=======================

Classifiers can make use of the :class:`recordlinkage.base.BaseClassifier` for
algorithms. ScitKit-learn based models may want
:class:`recordlinkage.adapters.SKLearnAdapter` as subclass as well.

.. autoclass:: recordlinkage.base.BaseClassifier
    :members:
    :inherited-members:

Probabilistic models can use the Fellegi and Sunter base class. This class is
used for the :class:`recordlinkage.ECMClassifier` and the
:class:`recordlinkage.NaiveBayesClassifier`.

.. autoclass:: recordlinkage.classifiers.FellegiSunter
    :members:
    :inherited-members:

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
    :members:
    :inherited-members:

.. autoclass:: recordlinkage.OneToManyLinking
    :members:
    :inherited-members:

.. autoclass:: recordlinkage.ConnectedComponents
    :members:
    :inherited-members:
