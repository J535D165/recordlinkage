# classifier.py

import pandas as pd
import numpy as np

import logging
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import em_algorithm_

from sklearn import cluster, linear_model, naive_bayes, svm

class Classifier(object):
	""" 

	Base class for classification of records pairs. This class contains methods for training the
	classifier. Distinguish different types of training, such as supervised and unsupervised
	learning.

	"""

	def __init__(self):

		self._params = {}

		# The actual classifier. Maybe this is slightly strange because of inheritance.
		self.classifier = None

	def learn(self, comparison_vectors, match_index=None):
		""" 

		Train the classifer. In case of supervised learning, the second argument can be used to
		label the matches (1) and non-matches (0).

		:param comparison_vectors: The dataframe with comparison vectors. 
		:param match_index: The index of the matches. 

		:type comparison_vectors: pandas.DataFrame
		:type match_index: pandas.MultiIndex

		:return: A pandas Series with the labels 1 (for the matches) and 0 (for the non-matches). 
		:rtype: pandas.Series

		"""

		raise NotImplementedError("Class %s has no method 'learn()' " % self.__name__)

	def predict(self, comparison_vectors):
		""" 

		Classify a set of record pairs based on their comparison vectors into matches, non-matches
		and possible matches. The classifier has to be trained to call this method. 

		:param comparison_vectors: The dataframe with comparison vectors. 
		:type comparison_vectors: pandas.DataFrame

		:return: A pandas Series with the labels 1 (for the matches) and 0 (for the non-matches). 
		:rtype: pandas.Series

		"""

		raise NotImplementedError("Class %s has no method 'predict()' " % self.__name__)

	def prob(self, comparison_vectors):
		""" 

		Estimate the probability for each record pairs of being a match.

		The method computes the probability for each given record pair of being a match. The
		probability of a non-match is 1 minus the result. This method is not implemented for all
		classifiers (for example K-means clustering).

		:param comparison_vectors: The dataframe with comparison vectors. 
		:type comparison_vectors: pandas.DataFrame

		:return: A pandas Series with pandas.MultiIndex with the probability of being a match. 
		:rtype: pandas.Series
		"""

		raise NotImplementedError("Class %s has no method 'prob()' " % self.__name__)

class DeterministicClassifier(Classifier):
	""" 

	Base class for deterministic classification of records pairs. This class contains methods for
	training the classifier. Distinguish different types of training, such as supervised and
	unsupervised learning.

	"""

	def __init__(self):

		self._params = {}

class ProbabilisticClassifier(Classifier):
	""" 

	Base class for probabilistic classification of records pairs. This class contains methods for
	training the classifier. Distinguish different types of training, such as supervised and
	unsupervised learning.

	"""

	def __init__(self):

		self._params = {}

class KMeansClassifier(Classifier):
	""" 

	The K-means clusterings algorithm to classify the given record pairs into matches and non-
	matches. 

	.. note::

	   There are way better methods for linking records than the k-means clustering algorithm.
	   However, this algorithm does not need trainings data and is useful to do an initial guess.

	"""

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = cluster.KMeans(n_clusters=2, n_init=1)

	def learn(self, comparison_vectors):
		""" 

		Train the K-means classifier. The K-means classifier is unsupervised and therefore does not
		need labels. The K-means classifier classifies the data into two sets of links and non-
		links. The starting point of the cluster centers are 0.05 for the non-matches and 0.95 for
		the matches.

		:param comparison_vectors: The dataframe with comparison vectors. 
		:type comparison_vectors: pandas.DataFrame

		:return: A pandas Series with the labels 1 (for the matches) and 0 (for the non-matches). 
		:rtype: pandas.Series

		"""
		# Set the start point of the classifier. 
		self.classifier.init = np.array([[0.05]*len(list(comparison_vectors)),[0.95]*len(list(comparison_vectors))])

		# print self.class
		self.classifier.fit(comparison_vectors.as_matrix())

		return self

	def predict(self, comparison_vectors):
		""" Predict the class for a set of comparison vectors. 

		After training the classifiers, this method can be used to classify comparison vectors for which the class is unknown. 

		:param comparison_vectors: The dataframe with comparison vectors. 
		:type comparison_vectors: pandas.DataFrame

		:return: A pandas Series with the labels 1 (for the matches) and 0 (for the non-matches). 
		:rtype: pandas.Series

		"""
		
		prediction = self.classifier.predict(comparison_vectors.as_matrix())

		return pd.Series(prediction, index=comparison_vectors.index, name='classification')

class LogisticRegressionClassifier(DeterministicClassifier):

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = linear_model.LogisticRegression()

	def learn(self, vectors, match_index):

		train_series = pd.Series(False, index=vectors.index)
		train_series.loc[match_index & vectors.index] = True

		self.classifier.fit(vectors.as_matrix(), np.array(train_series))

		return self

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())

		return vectors.index[prediction.astype(bool)]

	def prob(self, vectors):
		probs = self.classifier.predict_proba(vectors.as_matrix())

		return pd.Series(probs[0,:], index=vectors.index)

class BernoulliNBClassifier(ProbabilisticClassifier):

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = naive_bayes.BernoulliNB()

	def learn(self, vectors, match_index):

		train_series = pd.Series(False, index=vectors.index)
		train_series.loc[match_index & vectors.index] = True

		self.classifier.fit(vectors.as_matrix(), np.array(train_series))

		return self

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())

		return vectors.index[prediction.astype(bool)]

	def prob(self, vectors):
		probs = self.classifier.predict_proba(vectors.as_matrix())

		return pd.Series(probs[0,:], index=vectors.index)

class SVMClassifier(Classifier):

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = svm.LinearSVC()

	def learn(self, vectors, match_index):

		train_series = pd.Series(False, index=vectors.index)
		train_series.loc[match_index & vectors.index] = True

		self.classifier.fit(vectors.as_matrix(), np.array(train_series))

		return self

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())

		return vectors.index[prediction.astype(bool)]

class BernoulliEMClassifier(ProbabilisticClassifier):
	"""Expectation Maximisation classifier in combination with Fellegi and Sunter model"""
	
	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = em_algorithm_.ECMEstimate()

	def learn(self, vectors, params_init=None):

		# Default parameters
		if not params_init:
			params_init = {
				'p': 0.05,
				'm': {feature: {0: 0.1, 1:0.9} for feature in list(vectors)},
				'u': {feature: {0: 0.9, 1:0.1} for feature in list(vectors)}
			}
			
		self.classifier.p_init = params_init

		# Start training the classifier
		self.classifier.train(vectors)

		return self

	def predict(self, vectors, *args, **kwargs):

		return self.classifier.predict_proba(vectors.as_matrix(), *args, **kwargs)

	def prob(self, vectors):
		
		probs = self.classifier.predict_proba(vectors.as_matrix())

		return pd.Series(probs[0,:], index=vectors.index)

