from __future__ import division

from utils import check_type

import logging

import numpy as np
import pandas as pd

import itertools

class StandardSeries(pd.Series):
    """ A pandas Series like object with additional methods for standartising data. For example, functions to clean string data and numerical information. 
    
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    @check_type
    def clean(self, lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True, inplace=True):
        """
        clean(lower=True, replace_by_none='[^ \-\_A-Za-z0-9]+', replace_by_whitespace='[\-\_]', remove_brackets=True, inplace=True)
        
        Remove special tokens from a series of strings. 
    
        :param lower: Convert the strings in lowercase characters.
        :param replace_by_none: A regular expression that is replaced by ''.
        :param replace_by_whitespace: A regular expression that is replaced by a whitespace.
        :param inplace: If True, replace the current strings by their cleaned variant.
        :param remove_brackets: Remove all content between brackets and the brackets themselves. 

        :return: A cleaned Series of strings.
        :rtype: standartise.StandardSeries

        For example:

        >>> s = recordlinkage.StandardSeries(['Mary-ann', 'Bob :)', 'Angel', 'Bob (alias Billy)'])
        >>> print s.clean()
        mary ann
        bob
        angel
        bob

        """

        string = self if inplace else self.copy()

        # Lower the string if lower is True
        string = string.str.lower() if lower else string 

        # Remove all content between brackets
        if remove_brackets:
            string = string.str.replace(r'(\[.*?\]|\(.*?\)|\{.*?\})', '')

        # Remove the special characters
        string = string.str.replace(replace_by_none, '')
        string = string.str.replace(replace_by_whitespace, ' ')

        # Remove multiple whitespaces
        string = string.str.replace(r'\s\s+', ' ')

        # Strip string
        string = string.str.lstrip().str.rstrip()

        if not inplace:
            return string

    @check_type
    def phonenumbers(self, replace_by_none='[^0-9]+', inplace=True):
        """ Clean string formatted phonenumbers into string of intergers. 

        :return: A StandardSeries with cleaned phonenumbers.
        :rtype: standartise.StandardSeries
        """
        string = self if inplace else self.copy()
        string = string.astype(str)

        # Remove all special tokens
        string = string.str.replace(replace_by_none, '')

        # Convert string into integer
        return string.astype(str)

    @check_type
    def count_values(self):
        """DEPRECEATED SOON: Count the number of times a value occurs. The difference with pandas.value_counts is that this function returns the values for each row. 
    
        :return: A StandardSeries with value counts.
        :rtype: standartise.StandardSeries
        """

        logging.warning('This function is depreceated soon. Use value_occurence instead.')

        value_count = self.fillna(0)

        return StandardSeries(value_count.groupby(by=value_count.values).transform('count'))

    @check_type
    def value_occurence(self):
        """
        Count the number of times a value occurs. The difference with pandas.value_counts is that this function returns the values for each row. 

        :return: A StandardSeries with value counts.
        :rtype: standartise.StandardSeries
        """

        value_count = self.copy()
        value_count.fillna('NAN', inplace=True)

        return value_count.groupby(by=value_count.values).transform('count')

    # @check_type
    # def names(self, encode_method=None):
 
    #     if encode:
    #         return encode(encode_method)
    #     else:
    #         return self

    @check_type
    def encode(self, encode_method, inplace=True):

        try:
            import jellyfish
        except ImportError:
            print "Install jellyfish to use string encoding."

        if inplace:
            string = self
        else:
            string = self.copy()

        number_of_unique_values = len(string.unique())

        if encode_method == 'soundex':
            string.encode('unicode', inplace=True)
            return string.apply(lambda x: jellyfish.soundex(x))

        elif encode_method == 'nyiis':
            string.encode('unicode', inplace=True)
            return string.apply(lambda x: jellyfish.nyiis(x))

        elif encode_method == 'unicode':
            return  string.astype(unicode, raise_on_error=False)

        else:
            raise Exception("encoding method not found")

        print "Reduction ratio is %s" % (1-number_of_unique_values/len(self.unique()))

        if not inplace:
            return string

    @check_type
    def similar_values(self, threshold=0.8, inplace=True):
        """
        similar_values(threshold=0.8, inplace=True)

        Group strings with high similarities. 
    
        :param threshold: Two strings with similarity above this threshold are considered to be the same string. The threshold is a value equal or between 0 and 1. Default 0.8. 
        :param inplace: If True, replace the current strings by their cleaned variant. Default: True.

        :return: A Series of strings.
        :rtype: standartise.StandardSeries

        """
        try:
            import jellyfish
        except ImportError:
            print "Install jellyfish to use string encoding."

        if inplace:
            string = self
        else:
            string = self.copy()

        replace_tuples = []

        for pair in itertools.combinations(self[self.notnull()].astype(unicode).unique(), 2):

            sim = 1-jellyfish.levenshtein_distance(pair[0], pair[1])/np.max([len(pair[0]),len(pair[1])])

            if (sim >= threshold):
                replace_tuples.append(pair)

        # This is not a very clever solution I think. Don't known how to solve it atm: connected_components?
        for pair in replace_tuples: 

            if (sum(self == pair[0]) > sum(self == pair[1])):
                self = StandardSeries(self.str.replace(pair[1], pair[0]))
            else:
                self = StandardSeries(self.str.replace(pair[0], pair[1]))

        return StandardSeries(string)

    @property
    def _constructor(self):
        return StandardSeries

    @property
    def _constructor_expanddim(self):
        return StandardDataframe


class StandardDataFrame(pd.DataFrame):
    """ A pandas DataFrame like object with additional methods for standartising data. For example, functions to clean string data and numerical information. 
    
    """
    def __init__(self, *args, **kwargs):
        super(StandardDataFrame, self).__init__(*args, **kwargs)

    def names(self, columns, encoding=None, inplace=True):

    	columns = [columns] if type(columns) != list else list(columns)

    	for col in columns:

    		self[col] = self[col].str.lower()

    def clean_string(self, lowercase=True, remove_tokens=True, inplace=True):

    	for col in self.columns.tolist():

    		try:
    			self[col] = self[col].clean(inplace=False)
    		except Exception:
    			pass

    	return self


    @property
    def _constructor(self):
        return StandardDataFrame

    @property
    def _constructor_sliced(self):
        return StandardSeries

	def names(self, columns, encoding=None, inplace=True):

		columns = [columns] if type(columns) == str else list(columns)

		for col in columns: 

		    try:
		        self[col] = self[col].str.lower()
		        # self[col] = self[col].str.replace(r'.|\"|\'', "")
		        # self[col] = self[col].str.replace(r'-_', " ")
		        self[col] = self[col].str.replace(".", "")
		        self[col] = self[col].str.replace("-", " ")
		        self[col] = self[col].str.replace("\"", "")
		        self[col] = self[col].str.replace("\'", " ")
		        self[col] = self[col].str.replace("*", "")

		        
		    except Exception:
		        pass

		    if encoding: 

		    	self[col] = encode_names(columns, encoding=encoding)

		return self[columns]


	def encode_names(self, columns, encoding='soundex'):

		return self[columns]





 

