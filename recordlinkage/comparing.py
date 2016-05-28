from __future__ import division 

import sys

import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

try:
	import jellyfish
except ImportError:
	pass

from recordlinkage.utils import _label_or_column, _resample

class Compare(object):
	""" 

	Class to compare the attributes of candidate record pairs. The ``Compare`` class has several
	methods to compare data such as string similarity measures, numerical metrics and exact
	comparison methods.

	:param pairs: A MultiIndex of candidate record pairs. 
	:param df_a: The first dataframe. 
	:param df_b: The second dataframe.

	:type pairs: pandas.MultiIndex
	:type df_a: pandas.DataFrame
	:type df_b: pandas.DataFrame

	:returns: A compare class
	:rtype: recordlinkage.Compare

	:var pairs: The candidate record pairs.
	:var df_a: The first DataFrame.
	:var df_b: The second DataFrame.
	:var vectors: The DataFrame with comparison data.

	:vartype pairs: pandas.MultiIndex
	:vartype df_a: pandas.DataFrame
	:vartype df_b: pandas.DataFrame
	:vartype vectors: pandas.DataFrame

	:Example:

		In the following example, the record pairs of two historical datasets with census data are
		compared. The datasets are named ``census_data_1980`` and ``census_data_1990``. The
		``candidate_pairs`` are the record pairs to compare. The record pairs are compared on the
		first name, last name, sex, date of birth, address, place, and income.

		>>> comp = recordlinkage.Compare(
			candidate_pairs, census_data_1980, census_data_1990
			)
		>>> comp.fuzzy('first_name', 'name', method='jarowinkler')
		>>> comp.fuzzy('lastname', 'lastname', method='jarowinkler')
		>>> comp.exact('dateofbirth', 'dob')
		>>> comp.exact('sex', 'sex')
		>>> comp.fuzzy('address', 'address', method='levenshtein')
		>>> comp.exact('place', 'place')
		>>> comp.numerical('income', 'income')
		>>> print(comp.vectors.head())

		The attribute ``vectors`` is the DataFrame with the comparison data. It can be called whenever
		you want.

	"""

	def __init__(self, pairs, df_a=None, df_b=None):

		# The dataframes
		self.df_a = df_a
		self.df_b = df_b

		# The candidate record pairs
		self.pairs = pairs

		# The resulting data
		self.vectors = pandas.DataFrame(index=pairs)

		# self.ndim = self._compute_dimension(pairs)

	def compare(self, comp_func, data_a, data_b, *args, **kwargs):
		"""

		Core method to compare records. This method takes a function and data from both records in
		the record pair. The data is compared with the compare function. The built-in methods also
		use this function.

		:Example:

			Consider ``comp`` is a Compare instance. The code

			>>> comp.exact('first_name', 'name')

			is the same as

			>>> comp.compare(recordlinkage._compare_exact, 'first_name', 'name')

		:param comp_func: A comparison function. This function can be a built-in function or a user defined comparison function.
		:param data_a: The labels, Series or DataFrame to compare.
		:param data_b: The labels, Series or DataFrame to compare.
		:param name: The name of the feature and the name of the column.
		:param store: Store the result in the dataframe.

		:type comp_func: function
		:type data_a: label, pandas.Series, pandas.DataFrame
		:type data_b: label, pandas.Series, pandas.DataFrame
		:type name: label
		:type store: boolean, default True

		:return: The DataFrame Compare.vectors
		:rtype: standardise.DataFrame
		"""

		args = list(args)

		name = kwargs.pop('name', None)
		store = kwargs.pop('store', True)

		# Sample the data and add it to the arguments.
		if not isinstance(data_b, (tuple, list)):
			data_b = [data_b]

		if not isinstance(data_a, (tuple, list)):
			data_a = [data_a]

		for db in reversed(data_b):
			args.insert(0, _resample(_label_or_column(db, self.df_b), self.pairs, 1))

		for da in reversed(data_a):
			args.insert(0, _resample(_label_or_column(da, self.df_a), self.pairs, 0))

		c = comp_func(*tuple(args), **kwargs)

		# If it is a pandas Series, remove the name and replace it. 
		if isinstance(c, (pandas.DataFrame, pandas.Series)):
			c.name = name

		# Store the result the comparison result
		if store:
			self._append(c, name=name)

		return pandas.Series(c, index=self.pairs, name=name)

	def exact(self, s1, s2, *args, **kwargs):
		"""
		exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0, name=None, store=True)

		Compare the record pairs exactly.

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param agree_value: The value when two records are identical. Default 1. If 'values' is passed, then the value of the record pair is passed. 	
		:param disagree_value: The value when two records are not identical.
		:param missing_value: The value for a comparison with a missing value. Default 0.
		:param name: The name of the feature and the name of the column.
		:param store: Store the result in the dataframe.

		:type s1: label, pandas.Series
		:type s2: label, pandas.Series
		:type agree_value: numpy.dtype
		:type disagree_value: numpy.dtype
		:type missing_value: numpy.dtype
		:type name: label
		:type store: boolean, default True

		:return: A Series with comparison values.
		:rtype: pandas.Series

		"""

		return self.compare(_compare_exact, s1, s2, *args, **kwargs)

	def numerical(self, s1, s2, *args, **kwargs):
		"""
		numerical(s1, s2, window, missing_value=0, name=None, store=True)

		Compare numerical values with a tolerance window.

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param missing_value: The value for a comparison with a missing value. Default 0.
		:param window: The window size. Can be a tuple with two values or a single number. 
		:param name: The name of the feature and the name of the column.
		:param store: Store the result in the dataframe.

		:type name: label
		:type store: boolean, default True

		:return: A Series with comparison values.
		:rtype: pandas.Series

		"""

		return self.compare(_compare_numerical, s1, s2, *args, **kwargs)

	def fuzzy(self, s1, s2, *args, **kwargs):
		"""
		fuzzy(s1, s2, method='levenshtein', threshold=None, missing_value=0, name=None, store=True)

		Compare string values with a similarity approximation. 

		:param s1: Series or DataFrame to compare all fields. 
		:param s2: Series or DataFrame to compare all fields. 
		:param method: A approximate string comparison method. Options are ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'qgram', 'cosine']. Default: 'levenshtein'
		:param threshold: A threshold value. All approximate string comparisons higher or equal than this threshold are 1. Otherwise 0.  
		:param missing_value: The value for a comparison with a missing value. Default 0.
		:param name: The name of the feature and the name of the column.
		:param store: Store the result in the dataframe.

		:type name: label
		:type store: boolean, default True
		
		:return: A Series with similarity values. Values equal or between 0 and 1.
		:rtype: pandas.Series

		Note: For som of these algorithms is the package 'jellyfish' required. Install it with ``pip install jellyfish``.

		"""

		return self.compare(_compare_fuzzy, s1, s2, *args, **kwargs)

	def geo(self, x1, y1, x2, y2, *args, **kwargs):
		"""
		geo(x1, y1, x2, y2, radius=20, disagree_value = -1, missing_value=-1, name=None, store=True)

		[Experimental] Compare geometric coordinates with a tolerance window.

		:param x1: Series with X-coordinates
		:param y1: Series with Y-coordinates
		:param x2: Series with X-coordinates
		:param y2: Series with Y-coordinates
		:param missing_value: The value for a comparison with a missing value. Default -1.
		:param disagree_value: The value for a disagreeing comparison. Default -1.
		:param name: The name of the feature and the name of the column.
		:param store: Store the result in the dataframe.

		:type name: label
		:type store: boolean, default True
		
		:return: A Series with comparison values.
		:rtype: pandas.Series
		"""

		return self.compare(_compare_geo, (x1, y1), (x2, y2), *args, **kwargs)

	def batchcompare(self, list_of_comp_funcs):
		"""
		This method will be used to speed up the comparison of record pairs in the future.

		"""

		raise NotImplementedError("This method will be used to increase the comparing speed.")

	def _append(self, comp_vect, name=None):
		"""

		Add the comparison result to the ``vector`` attribute.

		"""

		if name:
			self.vectors[name] = comp_vect.values
		else:
			self.vectors[len(self.vectors)] = comp_vect

def _missing(*args):
	""" Internal function to return the index of record pairs with missing values """

	return np.any(np.concatenate([np.array(pandas.DataFrame(arg).isnull()) for arg in args], axis=1), axis=1)

def _compare_exact(s1, s2, agree_value=1, disagree_value=0, missing_value=0):

	if agree_value == 'value':
		compare = s1.copy()
		compare[s1 != s2] = disagree_value

	else:
		compare = np.where(s1 == s2, agree_value, disagree_value)
		
	compare = pandas.Series(compare, index=s1.index)

	# Only when disagree value is not identical with the missing value
	if disagree_value != missing_value:
		compare[_missing(s1, s2)] = missing_value

	return compare

def _compare_numerical(s1, s2, window, missing_value=0):

	if isinstance(window, (list, tuple)):
		compare = (((s1-s2) <= window[1]) & ((s1-s2) >= window[0])).astype(int)
	else:
		compare = (((s1-s2) <= window) & ((s1-s2) >= -window)).astype(int)

	compare[_missing(s1, s2)] = missing_value 

	return compare

def _compare_geo(x1, y1, x2, y2, radius=None, missing_value=np.nan):

	try:
		from scipy.spatial import distance
	except ImportError:
		raise ImportError('Could not import scipy.spatial')

	coord_1 = np.append([x1],[y1], axis=0).T
	coord_2 = np.append([x2],[y2], axis=0).T

	d = distance.cdist(coord_1, coord_2, 'cityblock')

	if not np.isnan(missing_value):
		d[_missing(d)] = missing_value

	return d 

def _check_jellyfish():

	if 'jellyfish' not in sys.modules:
		raise ImportError("Install the module 'jellyfish' to use the following string metrics: 'jaro', 'jarowinkler', 'levenshtein' and 'damerau_levenshtein'.")

def _compare_fuzzy(s1,s2, method='levenshtein', threshold=None, missing_value=0):

	if method == 'jaro':
		approx = jaro_similarity(s1, s2)

	elif method in ['jarowinkler', 'jaro_winkler']:
		approx = jarowinkler_similarity(s1, s2)

	elif method == 'levenshtein':
		approx = levenshtein_similarity(s1, s2)

	elif method in ['dameraulevenshtein', 'damerau_levenshtein']:
		approx = damerau_levenshtein_similarity(s1, s2)

	elif method in ['qgram', 'q_gram']:
		approx = qgram_similarity(s1, s2)

	elif method == 'cosine':
		approx = cosine_similarity(s1, s2)

	else:
		raise ValueError("""Algorithm '{}' not found.""".format(method))

	if threshold is not None:
		comp = (approx >= threshold).astype(int)
	else:
		comp = approx

	# Only for missing values
	comp[_missing(s1, s2)] = missing_value

	return comp

def jaro_similarity(s1,s2):

	# Check jellyfish
	_check_jellyfish()

	conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

	def jaro_apply(x):

		try:
			return jellyfish.jaro_distance(x[0],x[1])
		except Exception:
			return np.nan

	return conc.apply(jaro_apply, axis=1)

def jarowinkler_similarity(s1,s2):

	# Check jellyfish
	_check_jellyfish()
	
	conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

	def jaro_winkler_apply(x):

		try:
			return jellyfish.jaro_winkler(x[0],x[1])
		except Exception:
			return np.nan

	return conc.apply(jaro_winkler_apply, axis=1)

def levenshtein_similarity(s1,s2):

	# Check jellyfish
	_check_jellyfish()

	conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

	def levenshtein_apply(x):

		try:
			return 1-jellyfish.levenshtein_distance(x[0], x[1])/np.max([len(x[0]),len(x[1])])
		except Exception:
			return np.nan

	return conc.apply(levenshtein_apply, axis=1)

def damerau_levenshtein_similarity(s1,s2):

	# Check jellyfish
	_check_jellyfish()

	conc = pandas.concat([s1, s2], axis=1, ignore_index=True)

	def damerau_levenshtein_apply(x):

		try:
			return 1-jellyfish.damerau_levenshtein_distance(x[0], x[1])/np.max([len(x[0]),len(x[1])])
		except Exception:
			return np.nan

	return conc.apply(damerau_levenshtein_apply, axis=1)

def qgram_similarity(s1, s2, include_wb=True, ngram=(2,2)):

	if len(s1) != len(s2):
		raise ValueError('Arrays or Series have to be same length.')

	# include word boundaries or not
	analyzer = 'char_wb' if include_wb == True else 'char'

	# The vectorizer
	vectorizer = CountVectorizer(analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

	data = s1.append(s2).fillna('')

	vec_fit = vectorizer.fit_transform(data)
	
	def _metric_sparse_euclidean(u, v):
		match_ngrams = u.minimum(v).sum(axis=1)
		total_ngrams = np.maximum(u.sum(axis=1),v.sum(axis=1))

		return np.true_divide(match_ngrams,total_ngrams).A1

	return _metric_sparse_euclidean(vec_fit[:len(s1)], vec_fit[len(s1):])

def cosine_similarity(s1, s2, include_wb=True, ngram=(2,2)):

	if len(s1) != len(s2):
		raise ValueError('Arrays or Series have to be same length.')

	# include word boundaries or not
	analyzer = 'char_wb' if include_wb == True else 'char'

	# The vectorizer
	vectorizer = CountVectorizer(analyzer=analyzer, strip_accents='unicode', ngram_range=ngram)

	data = s1.append(s2).fillna('')

	vec_fit = vectorizer.fit_transform(data)

	def _metric_sparse_cosine(u,v):

		a = np.sqrt(u.multiply(u).sum(axis=1))
		b = np.sqrt(v.multiply(v).sum(axis=1))

		ab = v.multiply(u).sum(axis=1)

		return np.divide(ab, np.multiply(a,b)).A1

	return _metric_sparse_cosine(vec_fit[:len(s1)], vec_fit[len(s1):])


