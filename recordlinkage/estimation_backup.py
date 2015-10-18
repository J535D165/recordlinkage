
from __future__ import division

import time
import copy

import pandas as pd
import numpy as np

from sklearn.utils.extmath import cartesian

class EMEstimate(object):
    
    def __init__(self, start_m=None, start_u=None, start_p=None, comparison_vectors=None, comparison_space=None):
        
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

    def estimate(self, max_iter=100, log=False, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)


        print self.variables

        self.max_iter = max_iter
                
        while self.iteration < self.max_iter:
            
            # Compute expectation
            self.g = self._expectation(self.comparison_space[self.variables])

            print self.g
            
            # Maximize
            self.m, self.u, self.p = self._maximization(self.comparison_space[self.variables], self.comparison_space['count'], self.g)

            print self.m
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
        
        return np.log(np.divide(self.m_prob(y), self.u_prob(y)))
    
    def summary(self, include=None, exclude=None):
        
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

        # summary.set_index(self.variables) Add??

        return summary.reset_index()
    
    def cartesian(self): # aanpassen max. Moet unique worden..
        
        # Cartesian product of all possible options
        
        max_tuple = []
        
        for col, value_dict in self.m.items():

            max_tuple.append(value_dict.keys())
        #     # max_tuple.append(self.comparison_vectors[col].unique()) # Fix 
        
        y_cart = pd.DataFrame(cartesian(max_tuple)) # ([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1])
        y_cart.columns = self.variables

        return y_cart

    def add_comparison_vectors(self, comparison_vectors):

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

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
    
    def _maximization(self, y, f, g):
        
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
        
        return self.p*self.m_prob(y)/(self.p*self.m_prob(y)+(1-self.p)*self.u_prob(y)) 
    
    def m_prob(self, y):
        
        return y.replace(self.m).prod(axis=1)
    
    def u_prob(self, y):
        
        return y.replace(self.u).prod(axis=1)
