import pandas as pd
import numpy as np

import estimation

import logging


class Classifier(object):

	def __init__(self, *args, **kwargs):
		pass
	
	"""Get the matches of the comparison space. """
	def matches(self, comparison_vectors, w=None, p=None, mu=None):

		est_summary = self.est.summary()

		if w is not None:
			est_summary = est_summary[est_summary.weight >= w]
		
		if mu is not None:
			est_summary = est_summary[est_summary.mu <= mu]

		if p is not None:
			est_summary = est_summary[est_summary.p >= p]

		if w is None and mu is None and p is None:
			logging.warning('No thresholds set. All comparison vectors returned')

		return self._report(comparison_vectors, est_summary)

	"""Get the possible matches of the comparison space. """
	def possiblematches(self, comparison_vectors, w=None, p=None, lmbda=None, mu=None):

		est_summary = self.est.summary()

		if w is not None:
			est_summary = est_summary[(est_summary.weight > w[0]) & (est_summary.weight < w[1])]
		
		if lmbda is not None and mu is not None:
			est_summary = est_summary[(est_summary.lmbda > lmbda[0]) & (est_summary.mu < lmbda[1])]

		if p is not None:
			est_summary = est_summary[(est_summary.p > p[0]) & (est_summary.p < p[1])]

		if w is None and mu is None and p is None:
			logging.warning('No thresholds set. All comparison vectors returned')

		return self._report(comparison_vectors, est_summary)

	"""Get the non-matches of the comparison space. """
	def nonmatches(self, comparison_vectors, w=None, p=None, lmbda=None):

		est_summary = self.est.summary()

		if w is not None:
			est_summary = est_summary[est_summary.weight <= w]
		
		if lmbda is not None:
			est_summary = est_summary[est_summary['lambda'] <= lmbda]

		if p is not None:
			est_summary = est_summary[est_summary.p <= p]

		if w is None and lmbda is None and p is None:
			logging.warning('No thresholds set. All comparison vectors returned')

		return self._report(comparison_vectors, est_summary)

	def classify(self):
		""" To be overwritten """
		pass

class FellegiSunterClassifier(Classifier):
	"""Felllegi and Sunter method of classifying records into sets of matches, non-matches and possible matches. The work of Fellegi and Sunter is described in A theory of record linakge (Fellegi and Sunter, 1969). The classifier is based on likelihood ratios."""

	# Argument comparison_vectors is not nessesary the same as the comparison_vectors argument in
	# the estimation. (Use other comparison vectors to estimate than to classify)
	def __init__(self,  *args, **kwargs):
		super(self.__class__, self).__init__(*args, **kwargs)

		if 'est_method' not in kwargs.keys() and 'm' in kwargs.keys() and 'u' in kwargs.keys():
			# rename args
			self.est = estimation.ECMEstimate(*args, **kwargs)

	def estimate(self, est_method='ecm', *args, **kwargs):
		""" Estimate the parameters relevant for the Fellegi and Sunter framework. By default, the parameters are estimated with the Expectation/conditional Maximisation algorithm. The estimated parameters are arguments of FellegiSunterClassifier.est. Even if no the method estimate is not used, this is callable.

		:param est_method: Choose an estimation method. Currently, only one method is implemented. 

		"""

		if est_method in ['ecm', 'ci']: # for ci is no comparison space or vectors needed
			self.est = estimation.ECMEstimate()
			self.est.estimate(*args, **kwargs)
		else:
			print "The estimation method '%s'is not found." % est_method

	def ecm(self, *args, **kwargs):

		return self.estimate(est_method='ecm', *args, **kwargs)

	def _report(self, comparison_vectors, est_summary, sort=True):

		if sort:
			return comparison_vectors.merge(est_summary, on=self.est.variables, right_index=True).sort(columns='weight', ascending=False)
		else:
			return comparison_vectors.merge(est_summary, on=self.est.variables, right_index=True)

	def auto_classify(self, comparison_vectors, inplace=True):

		e = self.est.summary()

		col_status_name = 'status' 
		labels = ['M', 'P', 'U']

		e[col_status_name] = labels[2]
		e.loc[e['count'].cumsum() <= self.est.p*e['count'].sum(), col_status_name] = labels[0]

		if e[e['count'].cumsum() == self.est.p*e['count'].sum()].empty:
			e.loc[e['count'].cumsum() == self.est.p*e['count'].sum(), col_status_name] = labels[1]

		logging.info(e)

		return self._report(comparison_vectors, e)

class Deterministic:

	def __init__(self, comparison_vectors, ci_weights=None):
		
		self.comparison_vectors = comparison_vectors

		# if ci_weights and weights:
		# 	raise Exception('Warning: Use only conditional independent weights or normal weights.')

		self.ci_weights = ci_weights #self.set_ci_weights(ci_weights) if ci_weights else None
		# self.weights = self._ind_weights_to_weights(weights) if self.ci_weights else weights

		# """Set the weights for each comparison vectors. This is a pandas dataframe with comparison vectors and a weight column."""
		# def set_weights(self, weights):

		# 	self.weights = weights

		self.estimate = DeterministicEstimate(self)

	def set_ci_weights(self, ci_weights):

		if len(ci_weights) == len(self.comparison_vectors.columns.tolist()):
			self.ci_weights = ci_weights
		else:
			raise ValueError("Incorrect number of weights given.")

	def weight_vector(self):

		w_vectors = self.comparison_vectors.copy()

		for w_col in list(w_vectors):

			w_vectors[w_col] = (self.ci_weights[w_col][0] - self.ci_weights[w_col][1])*w_vectors[w_col]+self.ci_weights[w_col][1]
			
			# Substitute missing values. 
			w_vectors.fillna(self.ci_weights[w_col][2], inplace=True)

		self.vector_weights = w_vectors.sum(axis=1)

		return self.vector_weights

	# def classify(self, thresholds=None):

	# 	wv = self.weight_vector()

	# 	classes = []

	# 	thresholds.append(np.inf)

	# 	for th in range(1, len(thresholds)):

	# 		class_th = self.comparison_vectors[(wv <= thresholds[th]) & (wv > thresholds[th-1])].copy()
	# 		self.classes.append(class_th)

	# 	return classes

	"""Get the matches of the comparison space. """
	def matches(self, threshold):

		return self.comparison_vectors[self.weight_vector() >= threshold]

	"""Get the possible matches of the comparison space. """
	def possiblematches(self, threshold_low, threshold_high):

		return self.comparison_vectors[(self.weight_vector() > threshold_low) & (self.weight_vector() < threshold_high)]

	"""Get the non-matches of the comparison space. """
	def nonmatches(self, threshold):

		return self.comparison_vectors[self.weight_vector() <= threshold]

	# def _ind_weights_to_weights(self):

	# 	if self.ci_weights:

	# 		# Make comparison space is not avaible. 
	# 		if self.comparison_space is None:
	# 			self.comparison_space = self._make_comparison_space()

	# 		weight_space = self.comparison_space.copy()

	# 		for col in self.comparison_space.columns.tolist():


	# 			weight_space[col] = comparison_space[col]*self.cond_weights[col][0] # some function

	# 	else:
	# 		raise RuntimeError('Conditional independent weights not found.')

	# def _make_comparison_space(self):

	# 	self.comparison_space = pd.DataFrame({'count' : self.comparison_vectors.groupby(list(self.comparison_vectors)).size()}).reset_index()

	# 	return self.comparison_space

class DeterministicEstimate:

	def __init__(self, det_class):

		self.det_class = det_class

	# def train(self):

	# 	self.det_class.set_ci_weights({2:(3,-3,0),3:(5,-1,0),4:(2,-1,0)})

	# def unsupervised(self):

	# 	return 










