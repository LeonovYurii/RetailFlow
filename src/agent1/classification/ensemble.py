from pathlib import Path

import joblib
import numpy as np

from agent1.classification.tfidf_classifier import TfidfIntentClassifier
from agent1.classification.semantic_classifier import SemanticIntentClassifier


class IntentClassifierEnsemble:
    """Ensemble of TF-IDF and semantic intent classifiers."""

    def __init__(
        self,
        tfidf_params=None,
        semantic_params=None,
        tfidf_weight=0.5,
    ):
        if not 0.0 <= tfidf_weight <= 1.0:
            raise ValueError("tfidf_weight must be between 0 and 1.")

        self.tfidf_weight = tfidf_weight
        self.semantic_weight = 1.0 - tfidf_weight

        self.tfidf_params = tfidf_params or {}
        self.semantic_params = semantic_params or {}

        self.tfidf_classifier = TfidfIntentClassifier(
            **self.tfidf_params
        )

        self.semantic_classifier = SemanticIntentClassifier(
            **self.semantic_params
        )

    def fit(self, X, y):
        """
        Train both classifiers.

        The TF-IDF model is trained directly on text.
        The semantic model first converts text to embeddings,
        then trains Logistic Regression on those embeddings.
        """
        self.tfidf_classifier.fit(X, y)

        X_emb = self.semantic_classifier.encode(X)
        self.semantic_classifier.fit(X_emb, y)

        self._check_classes()

        return self

    def predict_proba(self, X):
        """
        Return ensemble probabilities for all intents.

        New text is embedded once for the semantic classifier.
        """
        self._check_classes()

        tfidf_proba = self.tfidf_classifier.predict_proba(X)

        X_emb = self.semantic_classifier.encode(X)
        semantic_proba = self.semantic_classifier.predict_proba(X_emb)

        ensemble_proba = (
            self.tfidf_weight * tfidf_proba
            + self.semantic_weight * semantic_proba
        )

        return ensemble_proba

    def predict(self, X):
        """Predict the most likely intent for each input."""
        probabilities = self.predict_proba(X)

        best_indices = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[best_indices]

    def predict_top_k(self, X, k=3):
        """
        Return the top-k intent predictions and ensemble scores
        for each input text.
        """
        if k < 1:
            raise ValueError("k must be at least 1.")

        probabilities = self.predict_proba(X)

        k = min(k, len(self.classes_))

        results = []

        for row in probabilities:
            top_indices = np.argsort(row)[::-1][:k]

            predictions = [
                {
                    "intent": self.classes_[idx],
                    "score": float(row[idx]),
                }
                for idx in top_indices
            ]

            results.append(predictions)

        return results

    def predict_with_confidence(self, X):
        """
        Return the best intent, its ensemble score,
        and the margin between the first and second choices.
        """
        probabilities = self.predict_proba(X)

        results = []

        for row in probabilities:
            sorted_indices = np.argsort(row)[::-1]

            best_idx = sorted_indices[0]

            best_score = float(row[best_idx])

            if len(sorted_indices) > 1:
                second_idx = sorted_indices[1]
                second_score = float(row[second_idx])
                second_intent = self.classes_[second_idx]
                margin = best_score - second_score
            else:
                second_score = None
                second_intent = None
                margin = None

            results.append(
                {
                    "intent": self.classes_[best_idx],
                    "score": best_score,
                    "margin": margin,
                    "second_intent": second_intent,
                    "second_score": second_score,
                }
            )

        return results

    def save(self, path):
        """Save the fitted ensemble to disk."""
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved ensemble from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Saved ensemble model not found: {path}"
            )

        model = joblib.load(path)

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is not an instance of {cls.__name__}."
            )

        return model

    def is_fitted(self):
        """Return True if both underlying classifiers appear fitted."""
        try:
            self._check_classes()
            return True
        except (AttributeError, ValueError):
            return False

    def _check_classes(self):
        """Ensure both classifiers use identical class ordering."""
        tfidf_classes = self.tfidf_classifier.classes_
        semantic_classes = self.semantic_classifier.classes_

        if not np.array_equal(
            tfidf_classes,
            semantic_classes,
        ):
            raise ValueError(
                "TF-IDF and semantic classifiers have different "
                "class labels or class ordering."
            )

    @property
    def classes_(self):
        """Intent labels in the same order as predict_proba columns."""
        return self.tfidf_classifier.classes_