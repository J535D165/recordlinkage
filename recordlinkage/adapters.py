
class SKLearnAdapter(object):

    # # sklearn classifier (or one that behaves like an sklearn classifier)
    # # make this an abstract attribute
    # self.kernel = None

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
        except NotFittedError:
            raise NotFittedError(
                "{} is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this method.".format(
                    type(self).__name__
                )
            )

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