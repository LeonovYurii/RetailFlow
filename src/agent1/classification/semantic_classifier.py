"""Semantic intent classifier using sentence embeddings and Logistic Regression."""

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class SemanticIntentClassifier:
    """Intent classifier based on sentence embeddings and Logistic Regression."""

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_iter=1000,
        device=None,
    ):
        self.model_name = model_name
        self.device = device

        # Load only when encode() is called.
        self.embedding_model = None

        self.classifier = LogisticRegression(
            max_iter=max_iter
        )

    def _load_embedding_model(self):
        """Load the sentence-transformer model when first needed."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

        return self.embedding_model


    def encode(self, X, batch_size=64):
        """Convert text into semantic embedding vectors."""
        embedding_model = self._load_embedding_model()

        return embedding_model.encode(
            list(X),
            batch_size=batch_size,
            show_progress_bar=True,
        )

    def fit(self, X_emb, y):
        """Train Logistic Regression on precomputed embeddings."""
        self.classifier.fit(X_emb, y)
        return self

    def predict(self, X_emb):
        """Predict intent labels from embeddings."""
        return self.classifier.predict(X_emb)

    def predict_proba(self, X_emb):
        """Return probabilities for every intent."""
        return self.classifier.predict_proba(X_emb)

    def __getstate__(self):
        """
        Exclude the SentenceTransformer model when serializing.
        """
        state = self.__dict__.copy()
        state["embedding_model"] = None
        return state

    def __setstate__(self, state):
        """
        Restore the classifier without loading the embedding model.
        """
        self.__dict__.update(state)

    @property
    def classes_(self):
        return self.classifier.classes_