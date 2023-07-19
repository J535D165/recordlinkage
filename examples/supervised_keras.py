"""Example: Supervised learning with Neural Networks."""


import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import layers
except ModuleNotFoundError as err:
    raise ModuleNotFoundError("Please upgrade tensorflow.") from err

import recordlinkage as rl
from recordlinkage.adapters import KerasAdapter
from recordlinkage.base import BaseClassifier
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


# Initialise the Keras.
class NNClassifier(KerasAdapter, BaseClassifier):
    """Neural network classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_dim=8, activation="relu"))
        model.add(layers.Dense(8, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.train.AdamOptimizer(0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.kernel = model


cl = NNClassifier()
cl.fit(X_data, links_true)

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
print(probs[0:10])
