from __future__ import division 

import pandas as pd
import numpy as np

from indexing import IndexError

class Compare(object):
	"""A class to make comparing of records fields easier. It can be used to compare fields of the record pairs. """

	def __init__(self, pairs, A=None, B=None):

		self.A = A
		self.B = B
		self.pairs = pairs

		self.vectors = pd.DataFrame(index=pairs)

		self.ndim = self._compute_dimension(pairs)

	def exact(self, s1, s2, *args, **kwargs):
		"""
		exact(s1, s2, missing_value=0, disagreement_value=0, output='any', return_agreement_values=False):

		Compare the record pairs exactly.

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param missing_value: The value for a comparison with a missing value. Default 0.
		:param output: Default 'any'. This holds only for comparing dataframes.
		:param return_agreement_values: If return_agreement_values is True, each agreeing comparison returns the value instead of 1. Default False.		

		:return: A Series with comparison values.
		:rtype: pandas.Series

		"""

		return self.compare(exact, s1, s2, *args, **kwargs)

	def numerical(self, s1, s2, *args, **kwargs):
		"""
		numerical(s1, s2, window, missing_value=0)

		Compare numerical values with a tolerance window.

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param missing_value: The value for a comparison with a missing value. Default 0.
		:param window: The window size. Can be a tuple with two values or a single number. 

		:return: A Series with comparison values.
		:rtype: pandas.Series

		"""

		return self.compare(window_numerical, s1, s2, *args, **kwargs)

	def fuzzy(self, s1, s2, *args, **kwargs):
		"""
		fuzzy(s1, s2, method='levenshtein', threshold=None, missing_value=0)

		Compare string values with a similarity approximation. 

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param method: A approximate string comparison method. Options are ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein']. Default: 'levenshtein'
		:param threshold: A threshold value. All approximate string comparisons higher or equal than this threshold are 1. Otherwise 0.  
		:param missing_value: The value for a comparison with a missing value. Default 0.

		:return: A Series with similarity values. Values equal or between 0 and 1.
		:rtype: pandas.Series

		Note: For this function is the package 'jellyfish' required. 

		"""

		return self.compare(fuzzy, s1, s2, *args, **kwargs)

	def geo(self, X1, Y1, X2, Y2, *args, **kwargs):
		"""
		geo(X1, Y1, X2, Y2, radius=20, disagreement_value = -1, missing_value=-1)

		Compare geometric coordinates with a tolerance window.

		:param X1: Series with X-coordinates
		:param Y1: Series with Y-coordinates
		:param X2: Series with X-coordinates
		:param Y2: Series with Y-coordinates
		:param missing_value: The value for a comparison with a missing value. Default -1.
		:param disagreement_value: The value for a disagreeing comparison. Default -1.

		:param compare_method: levestein

		:return: A Series with comparison values.
		:rtype: pandas.Series
		"""

		return self.compare(compare_geo, (X1, Y1), (X2, Y2), *args, **kwargs)

	def compare(self, comp_func, data_a, data_b, *args, **kwargs):
		"""Compare the records given. 

		:param comp_func: A comparison function. This function can be a built-in function or a user defined comparison function.

		:return: The DataFrame Compare.vectors
		:rtype: standardise.DataFrame
		"""

		args = list(args)
		name = kwargs.pop('name', None)
		store = kwargs.pop('store', True)

		print name

		# Sample the data and add it to the arguments.
		if not isinstance(data_b, (tuple, list)):
			data_b = [data_b]

		if not isinstance(data_a, (tuple, list)):
			data_a = [data_a]

		for db in reversed(data_b):
			args.insert(0, self._resample(self._getcol(db, self.B), 1))

		for da in reversed(data_a):
			args.insert(0, self._resample(self._getcol(da, self.A), 0))

		# print args
		c = comp_func(*tuple(args), **kwargs)

		# Add the comparison result
		self._append(c, name=name, store=store)

		return c

	def batchcompare(self, list_of_comp_funcs):

		raise NotImplementedError()

	def _append(self, comp_vect, name=None, store=True):

		if store:

			comp_vect.name = name
			self.vectors[name] = comp_vect

	def _getcol(self, label_or_column, dataframe):
		""" 
		This internal function is used to transform an index and a dataframe into a reindexed dataframe or series. If already a Series or DataFrame is passed, nothing is done. 
		"""
		try:
			return dataframe[label_or_column]

		except Exception:
			return label_or_column

	def _resample(self, s, level_i):

		data = s.ix[self.pairs.get_level_values(level_i)]
		data.index = self.pairs

		return data

	def _compute_dimension(self, pairs):
		'''
		Internal function to compute the dimension of the comparison. Deduplication returns 1 and linking 2 dataframes returns 2.
		'''

		# duplication
		if len(pairs.names) == 2 and pairs.names[0] == pairs.names[1]:
			return 1
		else:
			return len(pairs.names)

def _missing(*args):

	return np.any(np.concatenate([np.array(pd.DataFrame(arg).isnull()) for arg in args], axis=1), axis=1)

def exact(s1, s2, missing_value=0, disagreement_value=0, output='any', return_agreement_values=False):
	"""
	Compare two series or dataframes exactly on all fields. 
	"""

	df1 = pd.DataFrame(s1)
	df2 = pd.DataFrame(s2)

	# Only when one of the input variables is a DataFrame
	if len(list(df1)) > 1 or len(list(df2)) > 1:

		compare = pd.DataFrame([(df1[col1] == df2[col2]) for col2 in list(df2) for col1 in list(df1)]).T

		# Any of the agreeing comparisons
		if output == 'any':
			compare = compare.any(axis=1)
			compare = compare.astype(int)

		# Max of the comparisons
		elif output == 'max':
			compare = compare.astype(int)
			compare = compare.max(axis=1)

		# Sum of the comparison vectors
		elif output == 'sum':
			compare = compare.sum(axis=1)

		# Unknown method
		else:
			raise ValueError('Unknown output method.')

	else:

		if not return_agreement_values:
			compare = (s1 == s2)
			compare = compare.astype(int)
			compare.loc[(s1 != s2)] = disagreement_value 

		else:
			compare = s1.copy()
			compare.loc[(s1 != s2)] = disagreement_value

	# Only for missing values
	compare[_missing(df1, df2)] = missing_value

	return pd.Series(compare)

def window_numerical(s1, s2, window, missing_value=0):

	if isinstance(window, (list, tuple)):
		compare = (((s1-s2) <= window[1]) & ((s1-s2) >= window[0])).astype(int)
	else:
		compare = (((s1-s2) <= window) & ((s1-s2) >= window)).astype(int)

	compare[_missing(s1, s2)] = missing_value 

	return compare

def compare_geo(X1, Y1, X2, Y2, radius=None, missing_value=9):

	distance = np.sqrt(np.power(X1-X2,2)+np.power(Y1-Y2,2))

	comp = (distance <= radius).astype(int)

	comp[_missing(X1, X2)] = missing_value
	comp[_missing(Y1, Y2)] = missing_value

	return comp 

def fuzzy(s1,s2, method='levenshtein', threshold=None, missing_value=0):

	try:
		import jellyfish
	except ImportError:
		print "Install jellyfish to use approximate string comparison."

	series = pd.concat([s1, s2], axis=1)

	if method == 'jaro':
		approx = series.apply(lambda x: jellyfish.jaro_distance(x[0], x[1]) if pd.notnull(x[0]) and pd.notnull(x[1]) else np.nan, axis=1)
	
	elif method == 'jarowinkler':
		approx = series.apply(lambda x: jellyfish.jaro_winkler(x[0], x[1]) if pd.notnull(x[0]) and pd.notnull(x[1]) else np.nan, axis=1)
	
	elif method == 'levenshtein':
		approx = series.apply(lambda x: jellyfish.levenshtein_distance(x[0], x[1])/np.max([len(x[0]),len(x[1])]) if pd.notnull(x[0]) and pd.notnull(x[1]) else np.nan, axis=1)
		approx = 1 - approx

	elif method == 'damerau_levenshtein':
		approx = series.apply(lambda x: jellyfish.damerau_levenshtein_distance(x[0], x[1])/np.max([len(x[0]),len(x[1])]) if pd.notnull(x[0]) and pd.notnull(x[1]) else np.nan, axis=1)
		approx = 1 - approx

	else:
		raise ValueError('The method %s is not found.' % method)

	if threshold is not None:
		comp = (approx >= threshold).astype(int)
	else:
		comp = approx

	# Only for missing values
	comp[_missing(s1, s2)] = missing_value

	return comp

def window(s1, s2, window, missing_value=0, disagreement_value=0, sim_func=None):

	diff = s2-s1

	w = window if isinstance(window, (list, tuple)) else (window,window)

	if sim_func == None:
		sim = diff[(diff <= window[1]) & (diff >= window[0])]

	elif sim_func == 'linear':
		pass

	else:
		compare = (((s1-s2) <= window) & ((s1-s2) >= window)).astype(int)

	compare[_missing(s1, s2)] = missing_value 

	return compare

def bin_compare_geq(cat1, cat2, missing_value=0):

	# compare
	comp = (cat1 >= cat2).astype(int)
	comp.ix[_missing(cat1, cat2)] = missing_value 

	return comp

def compare_levels(s1, s2, freq, split=np.array([1, 5, 10, 20, 50, 100, np.inf])):

	comp = freq.copy()

	comp.fillna(0, inplace=True)
	comp.loc[(s1 != s2)] = 0

	for sp in range(0, len(split)-1):
		comp.loc[(comp < split[sp+1]) & (comp >= split[sp])] = split[sp]

	return comp
