'''Example: Supervised learning with the Naive Bayes algorithm.

'''

from __future__ import print_function

import numpy as np

import recordlinkage as rl
from recordlinkage.datasets import binary_vectors

# create a dataset with the following settings
n_pairs = 50000
n_matches = 7000
m_simulate = np.array([.94, .81, .85, .90, .99, .70, .56, .92])
u_simulate = np.array([.19, .23, .50, .11, .20, .14, .50, .09])

# Create the dataset and return the true links.
X_data, links_true = binary_vectors(
    n_pairs,  # the number of candidate links
    n_matches,  # the number of true links
    m=m_simulate,  # the m probabilities
    u=u_simulate,  # the u probabilities
    random_state=535,  # set seed
    return_links=True)  # return true links

# Initialise the Expectation-Conditional Maximisation classifier.
cl = rl.NaiveBayesClassifier()
cl.fit(X_data, links_true)

# Print the parameters that are trained (m, u and p). Note that the estimates
# are very good.
print("p probability P(Match):", cl.p)
print("m probabilities P(x_i=1|Match):", cl.m_probs)
print("u probabilities P(x_i=1|Non-Match):", cl.u_probs)
print("log m probabilities P(x_i=1|Match):", cl.log_m_probs)
print("log u probabilities P(x_i=1|Non-Match):", cl.log_u_probs)
print("Weights of features:", cl.log_weights)
print("Weights of features:", cl.weights)

# evaluate the model
links_pred = cl.predict(X_data)
print("Predicted number of links:", len(links_pred))

cm = rl.confusion_matrix(links_true, links_pred, total=len(X_data))
print("Confusion matrix:\n", cm)

# compute the F-score for this classification
fscore = rl.fscore(cm)
print('fscore', fscore)
recall = rl.recall(links_true, links_pred)
print('recall', recall)
precision = rl.precision(links_true, links_pred)
print('precision', precision)

# Predict the match probability for each pair in the dataset.
probs = cl.prob(X_data)
