"""Module to wrap external machine learning models."""

__all__ = ["SKLearnAdapter", "KerasAdapter"]


class SKLearnAdapter:
    """SciKit-learn adapter for record pair classification.

    SciKit-learn adapter for record pair classification with SciKit-learn
    models.
    """

    @property
    def classifier(self):
        # raise warning
        return self.kernel

    @classifier.setter
    def classifier(self, classifier):
        self.kernel = classifier

    def _predict(self, features):
        """Predict matches and non-matches.

        Parameters
        ----------
        features : numpy.ndarray
            The data to predict the class of.

        Returns
        -------
        numpy.ndarray
            The predicted classes.
        """

        from sklearn.exceptions import NotFittedError

        try:
            prediction = self.kernel.predict(features)
        except NotFittedError as err:
            raise NotFittedError(
                "{} is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this method.".format(type(self).__name__)
            ) from err

        return prediction

    def _fit(self, features, y=None):
        if y is None:  # unsupervised
            self.kernel.fit(features)
        else:
            self.kernel.fit(features, y)

    def _prob_match(self, features):
        """Compute match probabilities.

        Parameters
        ----------
        features : numpy.ndarray
            The data to train the model on.

        Returns
        -------
        numpy.ndarray
            The match probabilties.
        """

        # compute the probabilities
        probs = self.kernel.predict_proba(features)

        # get the position of match probabilities
        classes = list(self.kernel.classes_)
        match_class_position = classes.index(1)

        return probs[:, match_class_position]


class KerasAdapter:
    """Keras adapter for record pair classification.

    Keras adapter for record pair classification with Keras models.
    """

    @property
    def classifier(self):
        # raise warning
        return self.kernel

    @classifier.setter
    def classifier(self, classifier):
        self.kernel = classifier

    def _predict(self, features):
        """Predict matches and non-matches.

        Parameters
        ----------
        features : numpy.ndarray
            The data to predict the class of.

        Returns
        -------
        numpy.ndarray
            The predicted classes.
        """

        from sklearn.exceptions import NotFittedError

        try:
            prediction = self.kernel.predict_classes(features)[:, 0]
        except NotFittedError as err:
            raise NotFittedError(
                "{} is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this method.".format(type(self).__name__)
            ) from err

        return prediction

    def _fit(self, features, y=None):
        self.kernel.fit(features, y)

    def _prob_match(self, features):
        """Compute match probabilities.

        Parameters
        ----------
        features : numpy.ndarray
            The data to train the model on.

        Returns
        -------
        numpy.ndarray
            The match probabilties.
        """

        # compute the probabilities
        probs = self.kernel.predict_proba(features)[:, 0]

        return probs
