"""Example: Unsupervised learning with the ECM algorithm.

Train data is often hard to collect in record linkage or data matching
problems. The Expectation-Conditional Maximisation (ECM) algorithm is the most
well known algorithm for unsupervised data matching. The algorithm preforms
relatively well compared to supervised methods.

"""


import numpy as np

import recordlinkage as rl
from recordlinkage.datasets import binary_vectors

# create a dataset with the following settings
n_pairs = 50000
n_matches = 7000
m_simulate = np.array([0.94, 0.81, 0.85, 0.90, 0.99, 0.70, 0.56, 0.92])
u_simulate = np.array([0.19, 0.23, 0.50, 0.11, 0.20, 0.14, 0.50, 0.09])

# Create the dataset and return the true links.
X_data, links_true = binary_vectors(
    n_pairs,  # the number of candidate links
    n_matches,  # the number of true links
    m=m_simulate,  # the m probabilities
    u=u_simulate,  # the u probabilities
    random_state=535,  # set seed
    return_links=True,
)  # return true links

# Initialise the Expectation-Conditional Maximisation classifier.
cl = rl.ECMClassifier()
cl.fit(X_data)

# Print the parameters that are trained (m, u and p). Note that the estimates
# are very good.
print("p probability P(Match):", cl.p)
print("m probabilities P(x_i=1|Match):", cl.m_probs)
print("u probabilities P(x_i=1|Non-Match):", cl.u_probs)
print("log m probabilities P(x_i=1|Match):", cl.log_m_probs)
print("log u probabilities P(x_i=1|Non-Match):", cl.log_u_probs)
print("log weights of features:", cl.log_weights)
print("weights of features:", cl.weights)

# evaluate the model
links_pred = cl.predict(X_data)
print("Predicted number of links:", len(links_pred))

cm = rl.confusion_matrix(links_true, links_pred, total=len(X_data))
print("Confusion matrix:\n", cm)

# compute the F-score for this classification
fscore = rl.fscore(cm)
print("fscore", fscore)
recall = rl.recall(links_true, links_pred)
print("recall", recall)
precision = rl.precision(links_true, links_pred)
print("precision", precision)

# Predict the match probability for each pair in the dataset.
probs = cl.prob(X_data)
print(probs)
