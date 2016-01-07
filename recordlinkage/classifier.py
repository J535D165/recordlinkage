# classifier.py

import pandas as pd
import numpy as np

import logging

import em_algorithm_

from sklearn import cluster, linear_model

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

	def get_params(self, key):
		""" Get the parameters """
		self._params[key]

	def probabilities(self):
		""" Get the probability of being a true link for each comparison vector """

		raise AttributeError("class %s has no method 'probabilities()' " % self.__name__)

	def set_params(self, key, value):
		""" Set the parameters """

		self._params[key] = value

class KMeansClassifier(Classifier):
	""" 
	A clusterings algorithm to classify the given record pairs into matches and non-matches.
	"""
	def __init__(self, *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = cluster.KMeans(n_clusters=2)

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

		return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

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

		self.classifier = linear_model.LogisticRegression(n_clusters=2)

	def learn(self, vectors):

		# Set the start point of the classifier. 
		self.classifier.init = np.array([[0.05]*len(list(vectors)),[0.95]*len(list(vectors))])

		# print self.class
		self.classifier.fit(vectors.as_matrix())

		return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

	def predict(self, vectors):
		
		prediction = self.classifier.predict(vectors.as_matrix())

		return pd.Series(prediction, index=vectors.index, name='classification')

class ExpectationMaximisationClassifier(Classifier):
	"""Expectation Maximisation classifier in combination with Fellegi and Sunter model"""
	
	def __init__(self, method='ecm'):
		super(ExpectationMaximisationClassifier, self).__init__()

		self.method = method

		if method == 'ecm':
			self.classifier = em_algorithm_.ECMEstimate()
		else:
			raise ValueError("Method '%s' is unknown." % method)

	def learn(self, vectors, start_params=None):

		# If there are not start_params are passed, then apply standard starting params.
		if not start_params and self.method == 'ecm':
			start_parmas = {
				'm': [0.9]*len(list(vectors)),
				'u': [0.1]*len(list(vectors)),
				'pi': [0.05],				
			}

		self.classifier.init = start_params
		
		self.classifier.estimate(vectors)

		return pd.Series(self.classifier.labels_, index=vectors.index, name='classification')

	def predict(self, vectors):
		logging.warning('Be aware. Prediction can be risky. ')


