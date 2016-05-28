from __future__ import division

import pandas
import numpy

from recordlinkage.utils import IndexError
from recordlinkage.comparing import qgram_similarity

def _randomindex(df_a,df_b, n_pairs):

	if n_pairs <= 0 and type(n_pairs) is not int:
		raise ValueError("n_pairs must be an positive integer")

	if n_pairs < 0.25*len(df_a)*len(df_b):

		n_count = 0

		while n_count < n_pairs:

			random_index_a = numpy.random.choice(df_a.index.values, n_pairs-n_count)
			random_index_b = numpy.random.choice(df_b.index.values, n_pairs-n_count)

			sub_ind = pandas.MultiIndex.from_arrays([random_index_a, random_index_b], names=[df_a.index.name, df_b.index.name])

			ind = sub_ind if n_count == 0 else ind.append(sub_ind)
			ind = ind.drop_duplicates()

			n_count = len(ind)

		return ind

	else:

		full_index = _fullindex(df_a,df_b)

		return full_index[numpy.random.choice(numpy.arange(len(full_index)), n_pairs, replace=False)]

def _fullindex(df_a, df_b):

	return pandas.MultiIndex.from_product([df_a.index.values, df_b.index.values], names=[df_a.index.name, df_b.index.name])

def _blockindex(df_a, df_b, on=None, left_on=None, right_on=None):

	if on:
		left_on, right_on = on, on

	pairs = df_a[left_on].reset_index().merge(df_b[right_on].reset_index(), how='inner', left_on=left_on, right_on=right_on).set_index([df_a.index.name, df_b.index.name])

	return pairs.index

def _sortedneighbourhood(df_a, df_b, column, window=3, sorting_key_values=None, on=[], left_on=[], right_on=[]):

	# Check if window is an odd number
	if not bool(window % 2):
		raise ValueError('The given window length is not an odd integer.')

	# sorting_key_values is the terminology in Data Matching [Christen, 2012]
	if sorting_key_values is None:
		sorting_key_values = df_a[column].append(df_b[column])

	factors = pandas.Series(pandas.Series(sorting_key_values).unique())
	factors.sort_values(inplace=True)

	factors = factors[factors.notnull()].values # Remove possible numpy.nan values. They are not replaced in the next step.
	factors_label = numpy.arange(len(factors))

	sorted_df_A = pandas.DataFrame({column:df_a[column].replace(factors, factors_label), df_a.index.name: df_a.index.values})
	sorted_df_B = pandas.DataFrame({column:df_b[column].replace(factors, factors_label), df_b.index.name: df_b.index.values})

	pairs_concat = None

	# Internal window size
	_window = int((window-1)/2)

	for w in range(-_window, _window+1):

		pairs = sorted_df_A.merge(pandas.DataFrame({column:sorted_df_B[column]+w, df_b.index.name: df_b.index.values}), on=column, how='inner').set_index([df_a.index.name, df_b.index.name])

		# Append pairs to existing ones. PANDAS BUG workaround
		pairs_concat = pairs.index if pairs_concat is None else pairs.index.append(pairs_concat)

	return pairs_concat

def _qgram(df_a, df_b, on=None, left_on=None, right_on=None, threshold=0.8):

	fi = _fullindex(df_a, df_b)

	if on:
		left_on, right_on = on, on

	bool_index = (qgram_similarity(df_a.loc[fi.get_level_values(0), left_on], df_b.loc[fi.get_level_values(1), right_on]) >= threshold)

	return fi[bool_index]

class Pairs(object):
	""" 

	This class can be used to make record pairs. Multiple indexation methods can be used
	to make a smart selection of record pairs. Indexation methods included:

	- Full indexing
	- Blocking
	- Sorted Neighbourhood
	- Random indexing
	- Q-gram indexing

	For more information about indexing and methods to reduce the number of
	record pairs, see Christen 2012 [christen2012]_.

	:param df_a: The first dataframe. 
	:param df_b: The second dataframe.

	:type df_a: pandas.DataFrame
	:type df_b: pandas.DataFrame

	:returns: Candidate links
	:rtype: pandas.MultiIndex

	:var df_a: The first DataFrame.
	:var df_b: The second DataFrame.
	:var n_pairs: The number of candidate record pairs.

	:vartype df_a: pandas.DataFrame
	:vartype df_b: pandas.DataFrame
	:vartype n_pairs: int

	:Example:

		In the following example, the record pairs are made for two historical
		datasets with census data. The datasets are named ``census_data_1980``
		and ``census_data_1990``. 

		>>> pcl = recordlinkage.Pairs(census_data_1980, census_data_1990)
		>>> pcl.block('first_name')

	.. seealso::

		.. [christen2012] Christen, 2012. Data Matching Concepts and Techniques for 
					Record Linkage, Entity Resolution, and Duplicate Detection

	"""

	def __init__(self, df_a, df_b=None):

		self.df_a = df_a

		# Linking two datasets
		if df_b is not None:

			self.df_b = df_b
			self.deduplication = False

			if self.df_a.index.name == None or self.df_b.index.name == None:
				raise IndexError('DataFrame has no index name.')

			if self.df_a.index.name == self.df_b.index.name:
				raise IndexError("Identical index name '{}' for both dataframes.".format(self.df_a.index.name))

			if not self.df_a.index.is_unique or not self.df_b.index.is_unique:
				raise IndexError('DataFrame index is not unique.')

		# Deduplication of one dataset
		else:
			self.deduplication = True

			if self.df_a.index.name == None:
				raise IndexError('DataFrame has no index name.')

			if not self.df_a.index.is_unique:
				raise IndexError('DataFrame index is not unique.')

		self.n_pairs = 0

		self._index_factors = None

	# -- Index methods ------------------------------------------------------

	def index(self, index_func, *args, **kwargs):
		""" 

		Use a custom function to make record pairs of one or two dataframes.
		Each function should return a pandas.MultiIndex with record pairs.

		:param index_func: An indexing function
		:type index_func: function

		:return: MultiIndex
		:rtype: pandas.MultiIndex
		"""	

		# If not deduplication, make pairs of records with one record from the first dataset and one of the second dataset
		if not self.deduplication:

			pairs = index_func(self.df_a, self.df_b, *args, **kwargs)

		# If deduplication, remove the record pairs that are already included. For example: (a1, a1), (a1, a2), (a2, a1), (a2, a2) results in (a1, a2) or (a2, a1)
		elif self.deduplication:

			B = pandas.DataFrame(self.df_a, index=pandas.Index(self.df_a.index, name=str(self.df_a.index.name) + '_'))

			pairs = index_func(self.df_a, B, *args, **kwargs)

			# Remove all double pairs!
			pairs = pairs[pairs.get_level_values(0) < pairs.get_level_values(1)]
			pairs.names = [self.df_a, self.df_a]

		self.n_pairs = len(pairs)

		return pairs

	def full(self, *args, **kwargs):
		""" 
		full()

		Make an index with all possible record pairs. In case of linking two dataframes (A and B), the number of pairs is len(A)*len(B). In case of deduplicating a dataframe A, the number of pairs is len(A)*(len(A)-1)/2. 

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""
		return self.index(_fullindex, *args, **kwargs)

	def block(self, *args, **kwargs):
		""" 
		block(on=None, left_on=None, right_on=None)

		Make an index of record pairs agreeing on one or more specified attributes.

		:param on: A column name or a list of column names. These columns are used to block on. 
		:param left_on: A column name or a list of column names of dataframe A. These columns are used to block on. 
		:param right_on: A column name or a list of column names of dataframe B. These columns are used to block on. 

		:type on: label
		:type left_on: label
		:type right_on: label

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_blockindex, *args, **kwargs)

	def sortedneighbourhood(self, *args, **kwargs):
		"""
		sortedneighbourhood(sorting_key, window=3, sorting_key_values=None, on=[], left_on=[], right_on=[])

		Create a Sorted Neighbourhood index. 

		:param sorting_key: Specify the column to make a sorted index. 
		:param window: The width of the window, default is 3. 
		:param sorting_key_values: A list of sorting key values (optional).
		:param on: Additional columns to use standard blocking on. 
		:param left_on: Additional columns in the left dataframe to use standard blocking on. 
		:param right_on: Additional columns in the right dataframe to use standard blocking on. 

		:type sorting_key: label 
		:type window: int
		:type sorting_key_values: array
		:type on: label
		:type left_on: label
		:type right_on: label 

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""
		return self.index(_sortedneighbourhood, *args, **kwargs)

	def random(self, *args, **kwargs):
		""" 
		random(n_pairs)

		Make an index of randomly selected record pairs. 

		:param n_pairs: The number of record pairs to return. The integer n_pairs should satisfy 0 < n_pairs <= len(A)*len(B).

		:type n_pairs: int

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_randomindex, *args, **kwargs)


	def qgram(self, *args, **kwargs):
		""" 
		qgram(on=None, left_on=None, right_on=None, threshold=0.8)

		Use Q-gram string comparing metric to make an index.  

		:param on: A column name or a list of column names. These columns are used to index on. 
		:param left_on: A column name or a list of column names of dataframe A. These columns are used to index on. 
		:param right_on: A column name or a list of column names of dataframe B. These columns are used to index on. 
		:param threshold: Record pairs with a similarity above the threshold are candidate record pairs. [Default 0.8]

		:type on: label
		:type left_on: label
		:type right_on: label
		:type threshold: float

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_qgram, *args, **kwargs)

	# -- Iterative index methods ----------------------------------------------

	def iterindex(self, index_func, len_block_a=None, len_block_b=None, *args, **kwargs):
		"""Iterative function that returns records pairs based on a user-defined indexing function. The number of iterations can be adjusted to prevent memory problems.  

		:param index_func: A user defined indexing function.
		:param len_block_a: The length of a block of records in dataframe A. 
		:param len_block_b: The length of a block of records in dataframe B (only used when linking two datasets).

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""
		
		if self.deduplication:
			len_block_a = len_block_a if len_block_a else len(self.df_a) 
			
			blocks = [(a,a, a+len_block_a, a+len_block_a) for a in numpy.arange(0, len(self.df_a), len_block_a) for a in numpy.arange(0, len(self.df_a), len_block_a) ]

		else:
			len_block_a = len_block_a if len_block_a else len(self.df_a) 
			len_block_b = len_block_b if len_block_b else len(self.df_b) 
			
			blocks = [(a,b, a+len_block_a, b+len_block_b) for a in numpy.arange(0, len(self.df_a), len_block_a) for b in numpy.arange(0, len(self.df_b), len_block_b) ]

		# Reset the number of pairs counter
		self.n_pairs = 0

		for bl in blocks:

			if self.deduplication: # Deduplication
				pairs_block_class = Pairs(self.df_a[bl[0]:bl[2]], pandas.DataFrame(self.df_a, index=pandas.Index(self.df_a.index, name=self.df_a.index.name + '_')))
				pairs_block = pairs_block_class.index(index_func, *args, **kwargs)
				pairs_block = pairs_block[pairs_block.get_level_values(0) < pairs_block.get_level_values(1)]
	
			else:
				pairs_block_class = Pairs(self.df_a[bl[0]:bl[2]], self.df_b[bl[1]:bl[3]])
				pairs_block = pairs_block_class.index(index_func, *args, **kwargs)

			# Count the number of pairs
			self.n_pairs += len(pairs_block)
			
			yield pairs_block

	def iterfull(self, *args, **kwargs):
		"""
		iterfull(len_block_a=None, len_block_b=None)

		Iterative function that returns a part of a full index. 

		:param len_block_a: The length of a block of records in dataframe A. The integer len_block_a should satisfy 0 > len_block_a.
		:param len_block_b: The length of a block of records in dataframe B. The integer len_block_b should satisfy 0 > len_block_b.

		:type len_block_a: int
		:type len_block_b: int

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""
		return self.iterindex(_fullindex, *args, **kwargs)

	def iterblock(self, *args, **kwargs):
		"""
		iterblock(len_block_a=None, len_block_b=None, on=None, left_on=None, right_on=None)

		Iterative function that returns a part of a blocking index.

		:param len_block_a: The length of a block of records in dataframe A. The integer len_block_a should satisfy 0 > len_block_a.
		:param len_block_b: The length of a block of records in dataframe B. The integer len_block_b should satisfy 0 > len_block_b.
		:param on: A column name or a list of column names. These columns are used to block on. 
		:param left_on: A column name or a list of column names of dataframe A. These columns are used to block on. 
		:param right_on: A column name or a list of column names of dataframe B. These columns are used to block on. 

		:type len_block_a: int
		:type len_block_b: int
		:type on: label
		:type left_on: label
		:type right_on: label

		:param columns: A column name or a list of column names. These columns are used to block on. 

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""		
		return self.iterindex(_blockindex, *args, **kwargs)

	def itersortedneighbourhood(self, *args, **kwargs):
		"""
		itersortedneighbourhood(len_block_a=None, len_block_b=None, sorting_key, window=3, sorting_key_values=None, on=[], left_on=[], right_on=[])

		Iterative function that returns a records pairs based on a sorted neighbourhood index. The number of iterations can be adjusted to prevent memory problems.  

		:param len_block_a: The length of a block of records in dataframe A. The integer len_block_a should satisfy 0 > len_block_a.
		:param len_block_b: The length of a block of records in dataframe B. The integer len_block_b should satisfy 0 > len_block_b.
		:param sorting_key: Specify the column to make a sorted index. 
		:param window: The width of the window, default is 3. 
		:param sorting_key_values: A list of sorting key values (optional).
		:param on: Additional columns to use standard blocking on. 
		:param left_on: Additional columns in the left dataframe to use standard blocking on. 
		:param right_on: Additional columns in the right dataframe to use standard blocking on. 

		:type len_block_a: int
		:type len_block_b: int
		:type sorting_key: label 
		:type window: int
		:type sorting_key_values: array
		:type on: label
		:type left_on: label
		:type right_on: label 

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""
		column = args[2] # The argument after the two block size values

		# The unique values of both dataframes are passed as an argument. 
		sorting_key_values = numpy.sort(numpy.unique(numpy.append(self.df_a[column].values, self.df_b[column].values)))

		return self.iterindex(_sortedneighbourhood, *args, sorting_key_values=sorting_key_values, **kwargs)

	def iterqgram(self, *args, **kwargs):
		""" 
		iterqgram(len_block_a=None, len_block_b=None, on=None, left_on=None, right_on=None, threshold=0.8)

		Iterative function that returns Q-gram based index.  

		:param len_block_a: The length of a block of records in dataframe A. The integer len_block_a should satisfy 0 > len_block_a.
		:param len_block_b: The length of a block of records in dataframe B. The integer len_block_b should satisfy 0 > len_block_b.
		:param threshold: Record pairs with a similarity above the threshold are candidate record pairs.
		:param on: A column name or a list of column names. These columns are used to index on. 
		:param left_on: A column name or a list of column names of dataframe A. These columns are used to index on. 
		:param right_on: A column name or a list of column names of dataframe B. These columns are used to index on. 

		:type len_block_a: int
		:type len_block_b: int
		:type threshold: float
		:type on: label
		:type left_on: label
		:type right_on: label

		:return: The index of the candidate record pairs
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_qgram, *args, **kwargs)

	# -- Tools for indexing ----------------------------------------------

	@property
	def reduction(self):
		# """ 

		# The relative reduction of records pairs as the result of indexing. 

		# :param n_pairs: The number of record pairs.

		# :type n_pairs: int

		# :return: Value between 0 and 1
		# :rtype: float
		# """

		if self.deduplication:
			max_pairs = (len(self.df_a)*(len(self.df_b)-1))/2
		else:
			max_pairs = len(self.df_a)*len(self.df_b)

		return 1-self.n_pairs/max_pairs




