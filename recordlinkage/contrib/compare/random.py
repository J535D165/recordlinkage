"""Random compare strategy to test model behaviour."""

from numpy.random import random_sample
import pandas as pd

from recordlinkage.base import BaseCompareFeature


class RandomContinuous(BaseCompareFeature):
    """Add a feature with discrete random values.

    Parameters
    ----------
    a : float
        Lower bound of the continuous uniform distribution.
        Default 0.0.
    b : float
        Upper bound of the continuous uniform distribution.
        Default 1.0.
    label : list, str, int
        The identifying label(s) for the returned values.

    """

    name = "random"
    description = "Add a feature with only random values."

    def __init__(self,
                 a=0.0,
                 b=1.0,
                 label=None):
        super(RandomContinuous, self).__init__([], [], label=label)

        self.a = a
        self.b = b

    def _compute_vectorized(self, args, y):

        print(args.index.shape[0])

        random_values = random_sample(args.index.shape[0])

        if self.a != 0.0 or self.b != 1.0:
            random_values = (self.b - self.a) * random_values + self.a

        return random_values

    def compute(self, pairs, x=None, x_link=None):
        """Compare the records of each record pair.

        Parameters
        ----------
        pairs : pandas.MultiIndex
            A pandas MultiIndex with the record pairs to compare. The indices
            in the MultiIndex are indices of the DataFrame(s) to link.
        x : pandas.DataFrame
            The DataFrame to link. If `x_link` is given, the comparing is a
            linking problem. If `x_link` is not given, the problem is one of
            deduplication.
        x_link : pandas.DataFrame, optional
            The second DataFrame.

        Returns
        -------
        pandas.Series, pandas.DataFrame, numpy.ndarray
            The result of comparing record pairs (the features). Can be
            a tuple with multiple pandas.Series, pandas.DataFrame,
            numpy.ndarray objects.
        """

        df_empty = pd.DataFrame(index=pairs)
        return self._compute(
            tuple([df_empty]),
            tuple([df_empty])
        )
