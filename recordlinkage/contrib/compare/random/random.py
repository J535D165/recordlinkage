# Copyright 2018 Jonathan de Bruin
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Random compare strategy to test model behaviour."""

import numpy as np
import pandas as pd
from numpy.random import choice
from numpy.random import random_sample

from recordlinkage.base import BaseCompareFeature

__all__ = ["RandomContinuous", "RandomDiscrete"]


class RandomContinuous(BaseCompareFeature):
    """Add a feature with continuous random values.

    A column with continuous random values between 'a' and 'b' is
    returned. This comparison vector/feature can be useful for model
    testing.

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

    name = "random_cont"
    description = "Feature for continuous random values."

    def __init__(self, a=0.0, b=1.0, label=None):
        super().__init__([], [], label=label)

        self.a = a
        self.b = b

    def _compute_vectorized(self, args, y):
        random_values = random_sample(args.index.shape[0])

        if self.a != 0.0 or self.b != 1.0:
            random_values = (self.b - self.a) * random_values + self.a

        return random_values

    def compute(self, pairs, x=None, x_link=None):
        """Return continuous random values for each record pair.

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
        return self._compute(tuple([df_empty]), tuple([df_empty]))


class RandomDiscrete(BaseCompareFeature):
    """Add a feature with discrete random values.

    A column with discrete random values. This comparison vector/feature can
    be useful for model testing. By default, random values are sampled from
    a Bernoulli distribution with p=0.5.

    Parameters
    ----------
    a : int, numpy.ndarray
        If an ndarray, a random sample is generated from its
        elements. If an int, the random sample is generated
        as if a were np.arange(a). Default [0, 1]
    dtype : data-type
        The type of the data to return. Default np.int64.
    label : list, str, int
        The identifying label(s) for the returned values.

    """

    name = "random_desc"
    description = "Feature for discrete random values."

    def __init__(self, a=[0, 1], dtype=np.int64, label=None):
        super().__init__([], [], label=label)

        self.a = a
        self.dtype = dtype

    def _compute_vectorized(self, args, y):
        random_values = choice(self.a, args.index.shape[0])
        random_values = random_values.astype(self.dtype)

        return random_values

    def compute(self, pairs, x=None, x_link=None):
        """Return discrete random values for each record pair.

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
        return self._compute(tuple([df_empty]), tuple([df_empty]))
