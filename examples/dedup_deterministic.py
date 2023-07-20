"""Example: Deterministic record linkage to find links in a single file.

In determininistic record linkage, each compared attribute get a certain
weight (coefficient). The higher the weight, the more dicriminating the
variable is. A low weight indicate a less discriminating variable. For
example, the given name has a higher weight than the hometown.

This example uses FEBRL3 datasets. This dataset contain records about
individuals.

Deterministic RL parameters are:
intercept = -11.0
coefficients = [1.5, 1.5, 8.0, 6.0, 2.5, 6.5, 5.0]

"""


import recordlinkage as rl
from recordlinkage.compare import Exact
from recordlinkage.compare import String
from recordlinkage.datasets import load_febrl3
from recordlinkage.index import Block

# set logging
rl.logging.set_verbosity(rl.logging.INFO)

# load dataset
print("Loading data...")
dfA, true_links = load_febrl3(return_links=True)
print(len(dfA), "records in dataset A")
print(len(true_links), "links in dataset A")

# start indexing
print("Build index...")
indexer = rl.Index()
indexer.add(Block("given_name"))
indexer.add(Block("surname"))
indexer.add(Block("soc_sec_id"))
candidate_links = indexer.index(dfA)

# start comparing
print("Start comparing...")
comparer = rl.Compare()
comparer.add(Exact("given_name", "given_name", label="given_name"))
comparer.add(
    String("surname", "surname", method="jarowinkler", threshold=0.85, label="surname")
)
comparer.add(Exact("date_of_birth", "date_of_birth", label="date_of_birth"))
comparer.add(Exact("suburb", "suburb", label="suburb"))
comparer.add(Exact("state", "state", label="state"))
comparer.add(String("address_1", "address_1", threshold=0.85, label="address_1"))
comparer.add(String("address_2", "address_2", threshold=0.85, label="address_2"))
features = comparer.compute(candidate_links, dfA)

print("feature shape", features.shape)

# use the Logistic Regression Classifier
# this classifier is equivalent to the deterministic record linkage approach
intercept = -9.5
coefficients = [2.0, 3.0, 7.0, 6.0, 2.5, 5.0, 5.5]

print("Deterministic classifier")
print("intercept", intercept)
print("coefficients", coefficients)

logreg = rl.LogisticRegressionClassifier(coefficients=coefficients, intercept=intercept)
links = logreg.predict(features)

print(len(links), "links/matches")

# return the confusion matrix
conf_logreg = rl.confusion_matrix(true_links, links, len(candidate_links))
print("confusion matrix")
print(conf_logreg)

# compute the F-score for this classification
fscore = rl.fscore(conf_logreg)
print("fscore", fscore)
recall = rl.recall(true_links, links)
print("recall", recall)
precision = rl.precision(true_links, links)
print("precision", precision)
