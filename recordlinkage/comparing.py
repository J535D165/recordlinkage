from __future__ import division 

import pandas as pd
import numpy as np

from indexing import IndexError

class Compare(object):
	"""A class to make comparing of records fields easier. It can be used to compare fields of the record pairs. """

	def __init__(self, pairs=None, A=None, B=None):

		self.A = A
		self.B = B
		self.pairs = pairs

		self.vectors = pd.DataFrame(index=pairs)

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
		s1 = self._index_data(s1, dataframe=self.A)
		s2 = self._index_data(s2, dataframe=self.B)

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

		s1 = self._index_data(s1, dataframe=self.A)
		s2 = self._index_data(s2, dataframe=self.B)

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
		s1 = self._index_data(s1, dataframe=self.A)
		s2 = self._index_data(s2, dataframe=self.B)

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

		X1 = self._index_data(X1, dataframe=self.A)
		Y1 = self._index_data(Y1, dataframe=self.A)
		X2 = self._index_data(X2, dataframe=self.B)
		Y2 = self._index_data(Y2, dataframe=self.B)

		return self.compare(compare_geo, X1, Y1, X2, Y2, *args, **kwargs)

	def compare(self, comp_func, *args, **kwargs):
		"""Compare the records given. 

		:param comp_func: A list, dict or tuple of comparison tuples. Each tuple contains a comparison function, the fields to compare and the arguments. 

		:return: A DataFrame with compared variables.
		:rtype: standardise.DataFrame
		"""

		if isinstance(comp_func, (list, dict, tuple)):

			for t in comp_func:
				# func is a tuple of args and kwargs

				func = t[0]
				args_func = t[1]
				kwargs_func = t[2]

				self._append(self._compare_column(func, *args_func, **kwargs_func))

			return self.vectors

		else:

			name = kwargs.pop('name', None)
			store = kwargs.pop('store', True)

			c = comp_func(*args, **kwargs)
			self._append(c, name=name, store=store)

			if store:
				return self.vectors[name]
			else: 
				return c

	def _append(self, comp_vect, name=None, store=True):

		if store:

			comp_vect.name = name

			self.vectors[name] = np.array(comp_vect)

		return self.vectors

	def _index_data(self, *args, **kwargs):
		""" 
		This internal function is used to transform an index and a dataframe into a reindexed dataframe or series. If already a Series or DataFrame is passed, nothing is done. 
		"""

		dataframe = kwargs['dataframe']

		# # Check if dataframe name is not changed since making an index. Weird.
		# if dataframe.index.name not in self.pairs.names:
		# 	raise IndexError('The index name of the DataFrame is not found in the levels of the index.')

		if all(x in list(dataframe) for x in args):

			data = dataframe.ix[self.pairs.get_level_values(dataframe.index.name), args]
			data.set_index(self.pairs, inplace=True)

			return data[args[0]] if len(args) == 1 else (data[arg] for arg in args)
		
		else:
			# No labels passed, maybe series or dataframe passed? Let's try...
			# This is a trick to return tuples
			return args[0] if len(args) == 1 else args

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
