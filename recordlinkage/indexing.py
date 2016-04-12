from __future__ import division

import pandas
import numpy

def _randomindex(A,B, n_pairs):

	if n_pairs <= 0 and type(n_pairs) is not int:
		raise ValueError("n_pairs must be an positive integer")

	if n_pairs < 0.25*len(A)*len(B):

		n_count = 0

		while n_count < n_pairs:

			random_index_A = numpy.random.choice(A.index.values, n_pairs-n_count)
			random_index_B = numpy.random.choice(B.index.values, n_pairs-n_count)

			sub_ind = pandas.MultiIndex.from_arrays([random_index_A, random_index_B], names=[A.index.name, B.index.name])

			ind = sub_ind if n_count == 0 else ind.append(sub_ind)
			ind = ind.drop_duplicates()

			n_count = len(ind)

		return ind

	else:

		full_index = _fullindex(A,B)

		return full_index[numpy.random.choice(numpy.arange(len(full_index)), n_pairs, replace=False)]

def _fullindex(A, B):

	return pandas.MultiIndex.from_product([A.index.values, B.index.values], names=[A.index.name, B.index.name])

def _blockindex(A, B, on=None, left_on=None, right_on=None):

	if on:
		left_on, right_on = on, on

	pairs = A[left_on].reset_index().merge(B[right_on].reset_index(), how='inner', left_on=left_on, right_on=right_on).set_index([A.index.name, B.index.name])

	return pairs.index

def _sortedneighbourhood(A, B, column, window=3, sorting_key_values=None, on=[], left_on=[], right_on=[]):

	# Check if window is an odd number
	if not bool(window % 2):
		raise ValueError('The given window length is not an odd integer.')

	# sorting_key_values is the terminology in Data Matching [Christen, 2012]
	if sorting_key_values is None:
		sorting_key_values = A[column].append(B[column])

	factors = pandas.Series(pandas.Series(sorting_key_values).unique())
	factors.sort_values(inplace=True)

	factors = factors[factors.notnull()].values # Remove possible numpy.nan values. They are not replaced in the next step.
	factors_label = numpy.arange(len(factors))

	sorted_df_A = pandas.DataFrame({column:A[column].replace(factors, factors_label), A.index.name: A.index.values})
	sorted_df_B = pandas.DataFrame({column:B[column].replace(factors, factors_label), B.index.name: B.index.values})

	pairs_concat = None

	# Internal window size
	_window = int((window-1)/2)

	for w in range(-_window, _window+1):

		pairs = sorted_df_A.merge(pandas.DataFrame({column:sorted_df_B[column]+w, B.index.name: B.index.values}), on=column, how='inner').set_index([A.index.name, B.index.name])

		# Append pairs to existing ones. PANDAS BUG workaround
		pairs_concat = pairs.index if pairs_concat is None else pairs.index.append(pairs_concat)

	return pairs_concat

class Pairs(object):
	""" 

	Pairs class is used to make pairs of records to analyse in the comparison step. 

	"""	

	def __init__(self, dataframe_A, dataframe_B=None):

		self.A = dataframe_A

		# Linking two datasets
		if dataframe_B is not None:

			self.B = dataframe_B
			self.deduplication = False

			if self.A.index.name == None or self.B.index.name == None:
				raise IndexError('DataFrame has no index name.')

			if self.A.index.name == self.B.index.name:
				raise IndexError("Identical index name '{}' for both dataframes.".format(self.A.index.name))

			if not self.A.index.is_unique or not self.B.index.is_unique:
				raise IndexError('DataFrame index is not unique.')

		# Deduplication of one dataset
		else:
			self.deduplication = True

			if self.A.index.name == None:
				raise IndexError('DataFrame has no index name.')

			if not self.A.index.is_unique:
				raise IndexError('DataFrame index is not unique.')

		self.n_pairs = 0

		self._index_factors = None

	def index(self, index_func, *args, **kwargs):
		""" Method to make record pairs from one or two dataframes. Each record pair contains two records. 

		:return: MultiIndex
		:rtype: pandas.MultiIndex
		"""	

		# If not deduplication, make pairs of records with one record from the first dataset and one of the second dataset
		if not self.deduplication:

			pairs = index_func(self.A, self.B, *args, **kwargs)

		# If deduplication, remove the record pairs that are already included. For example: (a1, a1), (a1, a2), (a2, a1), (a2, a2) results in (a1, a2) or (a2, a1)
		elif self.deduplication:

			B = pandas.DataFrame(self.A, index=pandas.Index(self.A.index, name=str(self.A.index.name) + '_'))

			pairs = index_func(self.A, B, *args, **kwargs)

			# Remove all double pairs!
			pairs = pairs[pairs.get_level_values(0) < pairs.get_level_values(1)]
			pairs.names = [self.A, self.A]

		self.n_pairs = len(pairs)

		return pairs

	def random(self, *args, **kwargs):
		""" 
		random(A,B, n_pairs)

		Make an index of randomly selected record pairs. 

		:param A: The first DataFrame. 
		:param B: The second DataFrame.
		:param n_pairs: The number of record pairs to return. The integer n_pairs should satisfy 0 < n_pairs <= len(A)*len(B).

		:type A: pandas.DataFrame
		:type B: pandas.DataFrame
		:type n_pairs: int

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_randomindex, *args, **kwargs)

	def block(self, *args, **kwargs):
		""" 
		block(A, B, on=None, left_on=None, right_on=None)

		Make an index of record pairs agreeing on one or more specified attributes.

		:param A: The first DataFrame. 
		:param B: The second DataFrame.
		:param on: A column name or a list of column names. These columns are used to block on. 
		:param left_on: A column name or a list of column names of dataframe A. These columns are used to block on. 
		:param right_on: A column name or a list of column names of dataframe B. These columns are used to block on. 

		:type A: pandas.DataFrame
		:type B: pandas.DataFrame
		:type on: label
		:type left_on: label
		:type right_on: label

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""		
		return self.index(_blockindex, *args, **kwargs)

	def full(self, *args, **kwargs):
		""" 
		full(A, B)

		Make an index with all possible record pairs. In case of linking two dataframes (A and B), the number of pairs is len(A)*len(B). In case of deduplicating a dataframe A, the number of pairs is len(A)*(len(A)-1)/2. 

		:param A: The first DataFrame. 
		:param B: The second DataFrame.

		:type A: pandas.DataFrame
		:type B: pandas.DataFrame

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""
		return self.index(_fullindex, *args, **kwargs)

	def sortedneighbourhood(self, *args, **kwargs):
		"""
		sortedneighbourhood(A, B, sorting_key, window=3, sorting_key_values=None, on=[], left_on=[], right_on=[])

		Create a Sorted Neighbourhood index. 

		:param A: The first DataFrame. 
		:param B: The second DataFrame.
		:param sorting_key: Specify the column to make a sorted index. 
		:param window: The width of the window, default is 3. 
		:param sorting_key_values: A list of sorting key values (optional).
		:param on: Additional columns to use standard blocking on. 
		:param left_on: Additional columns in the left dataframe to use standard blocking on. 
		:param right_on: Additional columns in the right dataframe to use standard blocking on. 

		:type A: pandas.DataFrame
		:type B: pandas.DataFrame
		:type sorting_key: label 
		:type window: int
		:type sorting_key_values: array
		:type on: label
		:type left_on: label
		:type right_on: label 

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""
		return self.index(_sortedneighbourhood, *args, **kwargs)

	def iterblock(self, *args, **kwargs):
		"""Iterative function that returns a part of a blocking index.

		:param len_block_A: The lenght of a block of records in dataframe A. 
		:param len_block_B: The length of a block of records in dataframe B.
		:param columns: A column name or a list of column names. These columns are used to block on. 

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""		
		return self.iterindex(_blockindex, *args, **kwargs)

	def iterfull(self, *args, **kwargs):
		"""Iterative function that returns a part of a full index. 

		:param len_block_A: The lenght of a block of records in dataframe A. 
		:param len_block_B: The length of a block of records in dataframe B.
		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""
		return self.iterindex(_fullindex, *args, **kwargs)

	def itersortedneighbourhood(self, *args, **kwargs):
		"""Iterative function that returns a records pairs based on a sorted neighbourhood index. The number of iterations can be adjusted to prevent memory problems.  

		:param len_block_A: The lenght of a block of records in dataframe A. 
		:param len_block_B: The length of a block of records in dataframe B.
		:param column: Specify the column to make a sorted index. 
		:param window: The width of the window, default is 3. 
		:param suffixes: The suffixes to extend the column names with. 
		:param blocking_on: Additional columns to use standard blocking on. 
		:param left_blocking_on: Additional columns in the left dataframe to use standard blocking on. 
		:param right_blocking_on: Additional columns in the right dataframe to use standard blocking on. 

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""
		column = args[2] # The argument after the two block size values

		# The unique values of both dataframes are passed as an argument. 
		sorting_key_values = numpy.sort(numpy.unique(numpy.append(self.A[column].values, self.B[column].values)))

		return self.iterindex(_sortedneighbourhood, *args, sorting_key_values=sorting_key_values, **kwargs)

	def iterindex(self, index_func, len_block_A=None, len_block_B=None, *args, **kwargs):
		"""Iterative function that returns records pairs based on a user-defined indexing function. The number of iterations can be adjusted to prevent memory problems.  

		:param index_func: A user defined indexing funtion.
		:param len_block_A: The lenght of a block of records in dataframe A. 
		:param len_block_B: The length of a block of records in dataframe B (only used when linking two datasets).

		:return: A MultiIndex
		:rtype: pandas.MultiIndex
		"""

		# If block size is None, then use the full length of the dataframe
		
		if self.deduplication:
			len_block_A = len_block_A if len_block_A else len(self.A) 
			
			blocks = [(a,a, a+len_block_A, a+len_block_A) for a in numpy.arange(0, len(self.A), len_block_A) for a in numpy.arange(0, len(self.A), len_block_A) ]

		else:
			len_block_A = len_block_A if len_block_A else len(self.A) 
			len_block_B = len_block_B if len_block_B else len(self.B) 
			
			blocks = [(a,b, a+len_block_A, b+len_block_B) for a in numpy.arange(0, len(self.A), len_block_A) for b in numpy.arange(0, len(self.B), len_block_B) ]

		# Reset the number of pairs counter
		self.n_pairs = 0

		for bl in blocks:

			if self.deduplication: # Deduplication
				pairs_block_class = Pairs(self.A[bl[0]:bl[2]], pandas.DataFrame(self.A, index=pandas.Index(self.A.index, name=self.A.index.name + '_')))
				pairs_block = pairs_block_class.index(index_func, *args, **kwargs)
				pairs_block = pairs_block[pairs_block.get_level_values(0) < pairs_block.get_level_values(1)]
	
			else:
				pairs_block_class = Pairs(self.A[bl[0]:bl[2]], self.B[bl[1]:bl[3]])
				pairs_block = pairs_block_class.index(index_func, *args, **kwargs)

			# Count the number of pairs
			self.n_pairs += len(pairs_block)
			
			yield pairs_block

	def reduction(self, n_pairs=None):
		""" Compute the relative reduction of records pairs as the result of indexing. 

		:return: Value between 0 and 1
		:rtype: float
		"""

		if not n_pairs:
			n_pairs = self.n_pairs

		if self.deduplication:
			return self._reduction_ratio_deduplication(n_pairs)
		else:
			return self._reduction_ratio_linking(n_pairs)

	def _reduction_ratio_deduplication(self, n_pairs=None):

		max_pairs = (len(self.A)*(len(self.B)-1))/2

		return 1-self.n_pairs/max_pairs

	def _reduction_ratio_linking(self, n_pairs=None):

		max_pairs = len(self.A)*len(self.B)

		return 1-self.n_pairs/max_pairs

class IndexError(Exception):
	pass


