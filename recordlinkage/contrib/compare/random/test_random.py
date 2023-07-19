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

import pandas as pd

import recordlinkage
from recordlinkage.contrib.compare import RandomContinuous
from recordlinkage.contrib.compare import RandomDiscrete
from recordlinkage.index import Full


class TestRandomContinuous:
    def test_random_cont_standalone(self):
        arr1 = [1, 2, 3, 4, 5]
        arr2 = [1, 2, 3, 4, 5]
        pairs = pd.MultiIndex.from_product([arr1, arr2])

        c = RandomContinuous()
        r = c.compute(pairs)

        assert r.shape[0] == len(arr1) * len(arr2)

    def test_random_cont(self):
        df_a = pd.DataFrame({"v": list("abcde")})
        df_b = pd.DataFrame({"v": list("abcde")})

        pairs = Full().index(df_a, df_b)

        c = recordlinkage.Compare()
        c.exact("v", "v")
        c.add(RandomContinuous(label="random"))
        cv = c.compute(pairs, df_a, df_b)

        assert isinstance(cv, pd.DataFrame)

        assert cv["random"].notnull().all()
        assert cv["random"].min() >= 0.0
        assert cv["random"].max() <= 1.0


class TestRandomDiscrete:
    def test_random_desc_standalone(self):
        arr1 = [1, 2, 3, 4, 5]
        arr2 = [1, 2, 3, 4, 5]
        pairs = pd.MultiIndex.from_product([arr1, arr2])

        c = RandomDiscrete()
        r = c.compute(pairs)

        assert r.shape[0] == len(arr1) * len(arr2)

    def test_random_desc(self):
        df_a = pd.DataFrame({"v": list("abcde")})
        df_b = pd.DataFrame({"v": list("abcde")})

        pairs = Full().index(df_a, df_b)

        c = recordlinkage.Compare()
        c.exact("v", "v")
        c.add(RandomDiscrete(label="random"))
        cv = c.compute(pairs, df_a, df_b)

        assert isinstance(cv, pd.DataFrame)

        assert cv["random"].notnull().all()
        assert cv["random"].isin([0, 1]).all()
