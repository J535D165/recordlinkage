
from __future__ import division

import time
import copy

import pandas as pd
import numpy as np

from sklearn.utils.extmath import cartesian

class EMEstimate(object):
    
    def __init__(self, comparison_vectors=None, comparison_space=None, start_m=None, start_u=None, start_p=None):
        
        # Set first iteration
        self.m = start_m
        self.u = start_u
        self.p = start_p
        
        # Count the number of iterations
        self.iteration = 0

        # Set comparison vectors 
        if comparison_vectors is not None:
            # self.comparison_vectors = comparison_vectors
            self.comparison_space = self._group_comparison_vectors(comparison_vectors)
            self.variables = list(comparison_vectors)

        # Store comparison space
        if comparison_space is not None:
            self.comparison_space = comparison_space
            self.variables = [x for x in list(self.comparison_space) if x is 'count']

        self.number_of_pairs = self._number_of_pairs()
        
    def estimate(self, max_iter=100, *args, **kwargs):
        """Start the estimation of parameters with the iterative EM-algorithm. 

        :param max_iter: An integer specifying the maximum number of iterations. Default maximum number of iterations is 100. 
        """
        super(self.__class__, self).__init__(*args, **kwargs)

        self.max_iter = max_iter
                
        while self.iteration < self.max_iter:
            
            # Compute expectation
            self.g = self._expectation(self.comparison_space[self.variables])
            
            # Maximize
            self.m, self.u, self.p = self._maximization(self.comparison_space[self.variables], self.comparison_space['count'], self.g)

            # Increment counter
            self.iteration = self.iteration+1
        
    
    def _maximizion(self):
        
        """ To be overwritten """
        
        pass
    
    def _expectation(self):
        
        """ To be overwritten """
        
        pass

    def m_prob(self, y):
        
        """ To be overwritten """
        
        pass    

    def u_prob(self, y):
        
        """ To be overwritten """
        
        pass

    def weights(self, y):
        """Compute the weight for each comparison vector in the input dataframe. The weight is the log of the m- and u-probabilities. 

        :return: A Series with weights.
        :rtype: pandas.Series
        """

        return np.log(np.divide(self.m_prob(y), self.u_prob(y)))

    def summary(self, include=None, exclude=None):
        """Compute a summary report of all interesting parameters for each comparison vector in the comparison space. 

        :param include: A list of parameters to include in the summary for each comparison vectors. The default inlcuded parameters are 'm', 'u', 'weight', 'p_M', 'p', 'lambda', 'mu', 'count'. 
        :param exclude: A list of parameters to exclude in the summary for each comparison vectors.

        :return: A DataFrame with a cummary report for each compariosn vector.
        :rtype: pandas.DataFrame
        """

        summary = pd.merge(self.cartesian(), self.comparison_space, on=self.variables, how='left').fillna(0)

        include = ['m', 'u', 'weight', 'p_M', 'p', 'lambda', 'mu', 'count'] if include==None else include
        exclude = [] if exclude==None else exclude

        if 'm' in include and 'm' not in exclude:
            summary['m'] = self.m_prob(summary[self.variables])

        if 'u' in include and 'u' not in exclude:
            summary['u'] = self.u_prob(summary[self.variables])

        if 'weight' in include and 'weight' not in exclude:
            summary['weight'] = self.weights(summary[self.variables])

        if 'p_M' in include and 'p_M' not in exclude:
            summary['p_M'] = self._expectation(summary[self.variables])

        if 'p' in include and 'p' not in exclude:
            summary['p'] = self.p

        summary.sort('weight', ascending=True, inplace=True)
        if 'lambda' in include and 'lambda' not in exclude:
            summary['lambda'] = summary['m'].cumsum()

        summary.sort('weight', ascending=False, inplace=True)
        if 'mu' in include and 'mu' not in exclude:
            summary['mu'] = summary['u'].cumsum()

        return summary.reset_index(drop=True)

    def cartesian(self): # aanpassen max. Moet unique worden..
        """ Compute the cartesian product for the comparison vectors. This cartesian product contains all possible combinations of comparison vectors. 
        """
        
        # Cartesian product of all possible options
        max_tuple = []
        
        for col in self.variables:
            
            max_tuple.append(self.comparison_space[col].unique())
        
        y_cart = pd.DataFrame(cartesian(max_tuple)) # ([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
        y_cart.columns = self.variables

        return y_cart

    def add_comparison_vectors(self, comparison_vectors):
        """ Add a set of comparison vectors to the comparison space. This function is used to extend the comaprison space without having all comparison vectors stored in the internal memory. This is usefull when the record linkage is performed in batches of record pairs.  
        
        :param comparison_vectors: A DataFrame with comparison vectors. 

        """
        self.comparison_space = self._group_comparison_vectors(comparison_vectors)
        self.variables = [x for x in list(self.comparison_space) if x is 'count']

    def _group_comparison_vectors(self, comparison_vectors):

        return pd.DataFrame({'count' : comparison_vectors.groupby(list(comparison_vectors)).size()}).reset_index()

    def _number_of_pairs(self):

        try:
            return self.comparison_space['count'].sum()
        except Exception:
            return 0

class ECMEstimate(EMEstimate):     
    """ Algorithm to compute the Expectation/Conditional Maximisation algorithm in the context of record linkage. The algorithm is clearly described by Herzog, Schueren and Winkler in the book: Data Quality and Record Linkage Tehniques. The algorithm assumes that the comparison variables are mutually independent given the match status."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
    
    def _maximization(self, y, f, g):
        """ Maximisation step of the ECM-algorithm. 

        :param y: Dataframe with comparison vectors. 
        :param f: The number of times the comparison vectors y occur. This frame needs to have the same index as y. 
        :param g: The expectation of comparison vector in y.

        :return: A dict of marginal m-probabilities, a dict of marginal u-probabilities and the match prevalence. 
        :rtype: (dict, dict, float)
        """
        
        for col in y.columns:
            
            for level in y[col].unique():
                
                # Maximization of m
                self.m[col][level] = sum((g*f)[y[col] == level])/sum(g*f)
                                
                # Maximization of u
                self.u[col][level] = sum((((1-g)*f)[y[col] == level]))/sum((1-g)*f)
                
        # Maximization of p
        self.p = sum(g*f)/sum(f)
        
        return self.m, self.u, self.p
    
    def _expectation(self, y):
        """ Compute the expectation of the given comparison vectors. 

        :return: A Series with the expectation.
        :rtype: pandas.Series
        """
        
        return self.p*self.m_prob(y)/(self.p*self.m_prob(y)+(1-self.p)*self.u_prob(y)) 
    
    def m_prob(self, y):
        """Compute the m-probability, P(comparison vector|M), for a dataframe with comparison vectors. 

        :return: A Series with m-probabilities.
        :rtype: pandas.Series
        """

        y_m = y.copy()

        for col in set(self.variables) & set(self.m.keys()):

            keys, values = zip(*self.m[col].items())
            y_m[col].replace(keys, values, inplace=True)

        return y_m.prod(axis=1)
    
    def u_prob(self, y):
        """Compute the u-probability, P(comparison vector|U), for a dataframe with comparison vectors. 

        :return: A Series with u-probabilities.
        :rtype: pandas.Series
        """        
        y_u = y.copy()

        for col in set(self.variables) & set(self.u.keys()):

            keys, values = zip(*self.u[col].items())
            y_u[col].replace(keys, values, inplace=True)

        return y_u.prod(axis=1)
