
from __future__ import division

import time
import copy

import pandas as pd
import numpy

class EMEstimate(object):
    
    def __init__(self, max_iter=100, p_init=None, random_decisions=False):
        
        self.p_init = p_init
        self.max_iter = max_iter
        self.random_decisions = random_decisions

        self._p = None
        self._m = None
        self._u = None

        self._g = None

    def train(self, vectors, *args, **kwargs):
        """Start the estimation of parameters with the iterative EM-algorithm. 

        :param max_iter: An integer specifying the maximum number of iterations. Default maximum number of iterations is 100. 
        """

        if not self._m and not self._u and not self._p:
            self._m, self._u, self._p = self._quess_start_params(vectors)

        features_unique, feature_weights = self._count_vectors(vectors)

        self._iteration = 0
        
        while self._iteration < self.max_iter:
            
            # Expectation step
            g = self._expectation(features_unique)

            # Maximisation step
            self._maximization(features_unique, feature_weights, g)

            # Stop iterating when probs are close to previous iteration
            if self._g is not None and numpy.allclose(g, self._g, atol=10e-5):
                break

            self._g = g

            # Increment counter
            self._iteration += 1

    def _maximizion(self):
        
        """ Internal function to compute the maximisation step of the EM algorithm."""
        
        pass
    
    def _expectation(self):
        
        """ Internal function to compute the expectation step of the EM algorithm. """
        
        pass

    def _prob_m(self, y):
        
        """ Compute the m probabity of a DataFrame of comparison vectors. """
        
        pass    

    def _prob_u(self, y):
        
        """ Compute the u probabity of a DataFrame of comparison vectors. """
        
        pass

    def predict_proba(self, y):

        pass

    def _count_vectors(self, samples):

        samples_df = pd.DataFrame(samples)
        feature_combinations = samples_df.groupby(list(samples_df)).size()

        return (pd.DataFrame(index=feature_combinations.index).reset_index().as_matrix(), feature_combinations.values)

class ECMEstimate(EMEstimate):
    """ Algorithm to compute the Expectation/Conditional Maximisation algorithm in the context of record linkage. The algorithm is clearly described by Herzog, Schueren and Winkler in the book: Data Quality and Record Linkage Tehniques. The algorithm assumes that the comparison variables are mutually independent given the match status."""

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def _quess_start_params(self, samples):

        # Default parameters
        _p =  0.05
        _m = [{0:0.1, 1:0.9} for i in range(0, 10)]
        _u = [{0:0.9, 1:0.1} for i in range(0, 10)]

        return _m, _u, _p
    
    def _maximization(self, samples, weights, prob):
        """ Maximisation step of the ECM-algorithm. 

        :param samples: Dataframe with comparison vectors. 
        :param weights: The number of times the comparison vectors samples occur. This frame needs to have the same index as samples. 
        :param prob: The expectation of comparison vector in samples.

        :return: A dict of marginal m-probabilities, a dict of marginal u-probabilities and the match prevalence. 
        :rtype: (dict, dict, float)
        """

        for i in range(0, samples.shape[1]):
            
            for factor in numpy.unique(samples[:,i]):
                
                # Maximization of m
                self._m[i][factor] = numpy.sum((prob*weights)[samples[:,i] == factor])/numpy.sum(prob*weights)

                # Maximization of u
                self._u[i][factor] = numpy.sum((((1-prob)*weights)[samples[:,i] == factor]))/numpy.sum((1-prob)*weights)
        
        # Maximization of p
        self._p = numpy.sum(prob*weights)/numpy.sum(weights)
        
        return self._m, self._u, self._p
    
    def _expectation(self, samples):
        """ Compute the expectation of the given comparison vectors. 

        :return: A Series with the expectation.
        :rtype: pandas.Series
        """

        return self._p*self._prob_m(samples)/(self._p*self._prob_m(samples)+(1-self._p)*self._prob_u(samples)) 

    def _prob_m(self, samples):
        """Compute the m-probability, P(comparison vector|M), for a dataframe with comparison vectors. 

        :return: A Series with m-probabilities.
        :rtype: pandas.Series
        """

        newArray = numpy.zeros(samples.shape)

        for i in range(0, samples.shape[1]):

            for k in numpy.unique(samples[:,i]): 

                newArray[samples[:,i]==k, i] = self._m[i][k]

        return numpy.prod(newArray, axis=1)
    
    def _prob_u(self, samples):
        """Compute the u-probability, P(comparison vector|U), for a dataframe with comparison vectors. 

        :return: A Series with u-probabilities.
        :rtype: pandas.Series
        """

        newArray = numpy.zeros(samples.shape)

        for i in range(0, samples.shape[1]):

            for k in numpy.unique(samples[:,i]): 

                newArray[samples[:,i]==k, i] = self._u[i][k]

        return numpy.prod(newArray, axis=1)
       
    def predict_proba(self, samples):

        p_link = self._expectation(samples)
        p_nonlink = 1-p_link

        return numpy.array([p_link, p_nonlink])


