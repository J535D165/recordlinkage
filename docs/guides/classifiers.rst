Classification algorithms
=========================

In the context of record linkage, classification refers to the process
of dividing record pairs into matches and non-matches (distinct pairs).
There are dozens of classification algorithms for record linkage.
Roughly speaking, classification algorithms fall into two groups:

-  **supervised learning algorithms** - These algorithms make use of
   trainings data. If you do have trainings data, then you can use
   supervised learning algorithms. Most supervised learning algorithms
   offer good accuracy and reliability. Examples of supervised learning
   algorithms in the *Python Record Linkage Toolkit* are *Logistic
   Regression*, *Naive Bayes* and *Support Vector Machines*.
-  **unsupervised learning algorithms** - These algorithms do not need
   training data. The *Python Record Linkage Toolkit* supports *K-means
   clustering* and an *Expectation/Conditional Maximisation* classifier.


**First things first**

The examples below make use of the `Krebs
register <http://recordlinkage.readthedocs.org/en/latest/reference.html#recordlinkage.datasets.krebsregister_cmp_data>`__
(German for cancer registry) dataset. The Krebs register dataset
contains comparison vectors of a large set of record pairs. For each
record pair, it is known if the records represent the same person
(match) or not (non-match). This was done with a massive clerical
review. First, import the recordlinkage module and load the Krebs
register data. The dataset contains 5749132 compared record pairs and
has the following variables: first name, last name, sex, birthday, birth
month, birth year and zip code. The Krebs register contains
``len(krebs_true_links) == 20931`` matching record pairs.

.. ipython::

    In [0]: import pandas
       ...: import recordlinkage as rl
       ...: from recordlinkage.datasets import load_krebsregister

.. ipython::

    In [0]: krebs_X, krebs_true_links = load_krebsregister(missing_values=0)
       ...: krebs_X

Most classifiers can not handle comparison vectors with missing values.
To prevent issues with the classification algorithms, we convert the
missing values into disagreeing comparisons (using argument
missing\_values=0). This approach for handling missing values is widely
used in record linkage applications.

.. ipython::

    In [0]: krebs_X.describe()


Supervised learning
-------------------

As described before, supervised learning algorithms do need training
data. Training data is data for which the true match status is known for
each comparison vector. In the example in this section, we consider that
the true match status of the first 5000 record pairs of the Krebs
register data is known.

.. ipython::

    In [0]: golden_pairs = krebs_X[0:5000]
       ...: # 2093 matching pairs
       ...: golden_matches_index = golden_pairs.index.intersection(krebs_true_links)
       ...: golden_matches_index

Logistic regression
~~~~~~~~~~~~~~~~~~~

The ``recordlinkage.LogisticRegressionClassifier`` classifier is an
application of the logistic regression model. This supervised learning
method is one of the oldest classification algorithms used in record
linkage. In situations with enough training data, the algorithm gives
relatively good results.

.. ipython::

    In [0]: # Initialize the classifier
       ...: logreg = rl.LogisticRegressionClassifier()

    In [0]: # Train the classifier
       ...: logreg.fit(golden_pairs, golden_matches_index)
       ...: print ("Intercept: ", logreg.intercept)
       ...: print ("Coefficients: ", logreg.coefficients)


Predict the match status for all record pairs.

.. ipython::

    In [0]: result_logreg = logreg.predict(krebs_X)
       ...: len(result_logreg)


.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_logreg, len(krebs_X))

The F-score for this prediction is

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_logreg)


The predicted number of matches is not much more than the 20931 true
matches. The result was achieved with a small training dataset of 5000
record pairs.

In (older) literature, record linkage procedures are often divided in
**deterministic record linkage** and **probabilistic record linkage**.
The Logistic Regression Classifier belongs to deterministic record
linkage methods. Each feature/variable has a certain importance (named
weight). The weight is multiplied with the comparison/similarity vector.
If the total sum exceeds a certain threshold, it as considered to be a
match.

.. ipython::

    In [0]: intercept = -9
       ...: coefficients = [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]

    In [0]: logreg = rl.LogisticRegressionClassifier(coefficients, intercept)
       ...: result_logreg_pretrained = logreg.predict(krebs_X)
       ...: len(result_logreg_pretrained)

.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_logreg_pretrained, len(krebs_X))

The F-score for this classification is

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_logreg_pretrained)


For the given coefficients, the F-score is better than the situation
without trainings data. Surprising? No (use more trainings data and the
result will improve)

Naive Bayes
~~~~~~~~~~~

In contrast to the logistic regression classifier, the Naive Bayes
classifier is a probabilistic classifier. The probabilistic record
linkage framework by Fellegi and Sunter (1969) is the most well-known
probabilistic classification method for record linkage. Later, it was
proved that the Fellegi and Sunter method is mathematically equivalent
to the Naive Bayes method in case of assuming independence between
comparison variables.

.. ipython::

    In [0]: # Train the classifier
       ...: nb = rl.NaiveBayesClassifier(binarize=0.3)
       ...: nb.fit(golden_pairs, golden_matches_index)

.. ipython::

    In [0]: # Predict the match status for all record pairs
       ...: result_nb = nb.predict(krebs_X)
       ...: len(result_nb)


.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_nb, len(krebs_X))

The F-score for this classification is

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_nb)



Support Vector Machines
~~~~~~~~~~~~~~~~~~~~~~~

Support Vector Machines (SVM) have become increasingly popular in record
linkage. The algorithm performs well there is only a small amount of
training data available. The implementation of SVM in the Python Record
Linkage Toolkit is a linear SVM algorithm.

.. ipython::
    :okwarning:

    In [0]: # Train the classifier
       ...: svm = rl.SVMClassifier()
       ...: svm.fit(golden_pairs, golden_matches_index)

.. ipython::

    In [0]: # Predict the match status for all record pairs
       ...: result_svm = svm.predict(krebs_X)
       ...: len(result_svm)

.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_svm, len(krebs_X))


The F-score for this classification is

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_svm)


Unsupervised learning
---------------------

In situations without training data, unsupervised learning can be a
solution for record linkage problems. In this section, we discuss two
unsupervised learning methods. One algorithm is K-means clustering, and
the other algorithm is an implementation of the Expectation-Maximisation
algorithm. Most of the time, unsupervised learning algorithms take more
computational time because of the iterative structure in these
algorithms.

K-means clustering
~~~~~~~~~~~~~~~~~~

The K-means clustering algorithm is well-known and widely used in big
data analysis. The K-means classifier in the Python Record Linkage
Toolkit package is configured in such a way that it can be used for
linking records. For more info about the K-means clustering see
`Wikipedia <https://en.wikipedia.org/wiki/K-means_clustering>`__.

.. ipython::

    In [0]: kmeans = rl.KMeansClassifier()
       ...: result_kmeans = kmeans.fit_predict(krebs_X)
       ...: len(result_kmeans)

The classifier is now trained and the comparison vectors are classified.

.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_kmeans, len(krebs_X))

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_kmeans)


Expectation/Conditional Maximization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ECM-algorithm is an Expectation-Maximisation algorithm with some
additional constraints. This algorithm is closely related to the Naive
Bayes algorithm. The ECM algorithm is also closely related to estimating
the parameters in the Fellegi and Sunter (1969) framework. The
algorithms assume that the attributes are independent of each other. The
Naive Bayes algorithm uses the same principles.

.. ipython::

    In [0]: # Train the classifier
       ...: ecm = rl.ECMClassifier(binarize=0.8)
       ...: result_ecm = ecm.fit_predict(krebs_X)
       ...: len(result_ecm)

.. ipython::

    In [0]: rl.confusion_matrix(krebs_true_links, result_ecm, len(krebs_X))

The F-score for this classification is

.. ipython::

    In [0]: rl.fscore(krebs_true_links, result_ecm)
