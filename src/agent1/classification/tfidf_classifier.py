"""
TF-IDF based intent classifier.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfIntentClassifier:
    """Intent classifier based on TF-IDF features and Logistic Regression."""

    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_iter=1000,
    ):
        self.model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                ),
            ),
        ])

    def fit(self, X, y):
        """Train the intent classifier."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict intent labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return probabilities for every intent."""
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        """Intent labels in the same order as predict_proba columns."""
        return self.model.named_steps["classifier"].classes_