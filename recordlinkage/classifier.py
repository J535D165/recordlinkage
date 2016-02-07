# classifier.py

import pandas as pd
import numpy as np

import logging
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import em_algorithm_

from sklearn import cluster, linear_model
from sklearn import naive_bayes

class Classifier(object):
	""" Base class for classification of records pairs. This class contains methods for training the classifier. Distinguish different types of training, such as supervised and unsupervised learning."""

	def __init__(self):

		self._params = {}

		# The actual classifier. Maybe this is slightly strange because of inheritance.
		self.classifier = None

	def learn(self):
		""" Train the classifer. Sometimes this is done with training data."""
		pass

	def predict(self, comparison_vectors):
		""" Predict class of comparison vectors """
		pass
	
	# def get_params(self, key):
	# 	""" Get the parameters """
	# 	pass

	def prob(self):
		""" Get the probability of being a true link for each comparison vector """

		raise AttributeError("class %s has no method 'prob()' " % self.__name__)

	# def set_params(self, key, value):
	# 	""" Set the parameters """

	# 	self._params[key] = value

	# def false_positive_probability(self, vectors):
	# 	pass

	# def false_negative_probability(self, vectors):
	# 	pass

class KMeansClassifier(Classifier):
	""" 
	A clusterings algorithm to classify the given record pairs into matches and non-matches.
	"""
	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = cluster.KMeans(n_clusters=2, n_init=1)

	def learn(self, vectors):
		"""
		Train the classifier. No labels are needed. 

		:param vectors: A dataframe with comparison vectors.  

		:return: A pandas Series with the labels 1:matches and 0:non-matches.
		:rtype: pandas.Series

		"""
		# Set the start point of the classifier. 
		self.classifier.init = np.array([[0.05]*len(list(vectors)),[0.95]*len(list(vectors))])

		# print self.class
		self.classifier.fit(vectors.as_matrix())

		# return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

	def predict(self, vectors):
		"""
		Predict the class of a set of comparison vectors. Training the classifier first is required.

		:param vectors: A dataframe with comparison vectors.  

		:return: A pandas Series with the labels 1:matches and 0:non-matches.
		:rtype: pandas.Series

		"""		
		prediction = self.classifier.predict(vectors.as_matrix())

		return pd.Series(prediction, index=vectors.index, name='classification')

class LogisticRegressionClassifier(Classifier):

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = linear_model.LogisticRegression()

	def learn(self, vectors, match_index):

		train_series = pd.Series(False, index=vectors.index)
		train_series.loc[match_index & vectors.index] = True

		# print self.class
		self.classifier.fit(vectors.as_matrix(), np.array(train_series))

		# return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())
		prediction_bool = prediction.astype(bool)

		return vectors.index[prediction_bool], vectors.index[~prediction_bool]

	def prob(self, vectors, column_labels=['prob_link', 'prob_nonlink']):
		probs = self.classifier.predict_proba(vectors.as_matrix())

		return pd.DataFrame(probs, columns=column_labels, index=vectors.index)

class BernoulliNBClassifier(Classifier):

	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = naive_bayes.BernoulliNB()

	def learn(self, vectors, match_index):

		train_series = pd.Series(False, index=vectors.index)
		train_series.loc[match_index & vectors.index] = True

		# print self.class
		self.classifier.fit(vectors.as_matrix(), np.array(train_series))

		# return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())
		prediction_bool = prediction.astype(bool)

		return vectors.index[prediction_bool], vectors.index[~prediction_bool]

	def prob(self, vectors, column_labels=['prob_link', 'prob_nonlink']):
		probs = self.classifier.predict_proba(vectors.as_matrix())

		return pd.DataFrame(probs, columns=column_labels, index=vectors.index)

class ExpectationMaximisationClassifier(Classifier):
	"""Expectation Maximisation classifier in combination with Fellegi and Sunter model"""
	
	def __init__(self, method='ecm', random_decisions=False, p_init=None, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.method = method
		self.random_decisions = random_decisions
		self.p_init = p_init

		if method == 'ecm':
			self.classifier = em_algorithm_.ECMEstimate()
		else:
			raise ValueError("Method '%s' is unknown." % method)

	def learn(self, vectors):

		# No initial parameters
		if self.p_init==None:

			# The chosen method is ecm
			if self.method == 'ecm':

				default_params = {
					'p': 0.05,
					'm': {feature: {0: 0.1, 1:0.9} for feature in list(vectors)},
					'u': {feature: {0: 0.9, 1:0.1} for feature in list(vectors)}
				}

		self.classifier.p_init = default_params

		# Start training the classifier
		self.classifier.estimate(vectors)

		# Compute
		return self._classify(vectors)

	def predict(self, vectors, *args, **kwargs):

		return self._classify(vectors, *args, **kwargs)

	def prob(self, vectors, *args, **kwargs):

		return self._classify(vectors, *args, **kwargs)

	def _classify(self, vectors, p_match=None, p_nonmatch=None):

		vectors_unique, vector_counts = self.classifier._count_vectors(vectors)

		prob_vectors_unique = self.classifier._expectation(vectors_unique)

		N_match = np.floor(len(vectors)*self.classifier.p_init['p'])

		if not (vector_counts.sort_values(ascending=True).cumsum() == N_match).any():
			p_match = prob_vectors_unique[prob_vectors_unique > p_match]

		if not (vector_counts.cumsum(ascending=False) == N_match).any():
			p_nonmatch = prob_vectors_unique[prob_vectors_unique > p_nonmatch].tail(1)

		p_vectors = self.classifier._expectation(vectors)

		prediction = (p_vectors > p_match).astype(int)
		prediction.loc[(p_vectors < p_match) & (p_vectors > p_nonmatch)] = 9 # Unknown

		return pd.Series(prediction, index=vectors.index, name='classification')
