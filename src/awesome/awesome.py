import numpy as np
import sklearn.base
import sklearn.utils.validation


class AwesomeClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, i=0):
        self.i = i

    def fit(self, X, y=None):
        # Validate input according to sklearn's rules
        X, y = sklearn.utils.validation.check_X_y(X, y)

        # Store the classes seen during fit, as expected by sklearn
        self.X_ = X
        self.y_ = y

        # Store the number of features seen during fit, as expected by sklearn
        self.n_features_in_ = X.shape[1]

        # Store the classes seen during fit, as expected by sklearn
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)

        # Store the chosen label for our pseudo-classifier
        self.chosen_label_ = self.y_[self.i]

        # Return the classifier, as expected by sklearn
        return self

    def predict(self, X):
        # Ensure fit has been called, as expected by sklearn
        sklearn.utils.validation.check_is_fitted(self)

        # Validate input according to sklearn's rules
        X = sklearn.utils.validation.check_array(X)

        # Ensure the number of features is the same as during fit, as expected by sklearn
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has wrong number of features. Expected self.n_features_in_, got {X.shape[1]}."
            )

        # For each row in X, return label chosen during fit
        return np.full(X.shape[0], self.chosen_label_, dtype=self.classes_.dtype)

    def _more_tags(self):
        return {
            "poor_score": True,
            "_xfail_checks": {
                "check_classifiers_classes": "Fails because we always predict the same label."
            },
        }
