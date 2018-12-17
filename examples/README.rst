Python Record Linkage Toolkit examples
======================================

This folder contains examples on record linkage with the Python Record Linkage
Toolkit. The examples do have a BSD 3-Clause "New" or "Revised" License. 

Basic
-----

`Deterministic deduplication`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example of deterministic record linkage to find duplicated records in a
dataset. In this example, the model isn't trained with train data.

`Deterministic linkage`_
~~~~~~~~~~~~~~~~~~~~~~~~

Example of deterministic record linkage to find links between two datasets. In
this example, the model isn't trained with train data.

`Supervised Fellegi and Sunter (Naive Bayes)`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An implementation of the Fellegi and Sunter (1969) classification model in a
supervised way.

`Unsupervised Fellegi and Sunter (ECM)`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An implementation of the Fellegi and Sunter (1969) classification model in a
unsupervised way. The training of model parameters is done with the
Expectation-Contitional Maximisation algorithm.


Advanced
--------

`Record linkage with Neural Networks`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how Neural Networks can be used to classify record pairs.
The Neural Network is implemented in Keras.

.. _`Deterministic deduplication`: /examples/dedup_deterministic.py
.. _`Deterministic linkage`: /examples/linking_deterministic.py
.. _`Record linkage with Neural Networks`: /examples/supervised_keras.py
.. _`Supervised Fellegi and Sunter (Naive Bayes)` /examples/supervised_learning_prob.py
.. _`Unsupervised Fellegi and Sunter (ECM)` /examples/unsupervised_learning_prob.py
