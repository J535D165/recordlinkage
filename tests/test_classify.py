import unittest


import recordlinkage
from recordlinkage.datasets import load_krebsregister


class TestClassify(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.y, self.match_index = load_krebsregister(1)
        self.y.fillna(0, inplace=True)


    def test_kmeans(self):

        train_df = self.y.ix[self.match_index].sample(500)
        train_df = train_df.append(self.y.ix[self.y.index.difference(self.match_index)].sample(1500))

        kmeans = recordlinkage.KMeansClassifier()
        kmeans.learn(train_df)
        kmeans.predict(self.y)

    def test_logistic(self):

        train_df = self.y.ix[self.match_index].sample(500)
        train_df = train_df.append(self.y.ix[self.y.index.difference(self.match_index)].sample(1500))

        logis = recordlinkage.LogisticRegressionClassifier()
        logis.learn(train_df, self.match_index)
        logis.predict(self.y)

    def test_bernoulli_naive_bayes(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        bernb = recordlinkage.NaiveBayesClassifier()
        bernb.learn(train_df.round(), train_df_matches.index)
        bernb.predict(self.y.round())

    def test_svm(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        svm = recordlinkage.SVMClassifier()
        svm.learn(train_df.round(), train_df_matches.index)
        svm.predict(self.y.round())

    def test_em(self):

        train_df_matches = self.y.ix[self.match_index].sample(500)
        train_df_nonmatches = self.y.ix[self.y.index.difference(self.match_index)].sample(1500)
        train_df = train_df_matches.append(train_df_nonmatches)

        ecm = recordlinkage.ECMClassifier()
        ecm.learn(train_df.round())
        ecm.predict(self.y.round())
        ecm.prob(self.y.round())

        self.assertTrue(ecm.p is not None)
