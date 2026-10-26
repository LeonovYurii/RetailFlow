import json
from pathlib import Path


class KnowledgeStore:
    def __init__(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as f:
            collection = json.load(f)

        self.documents = collection["documents"]

    def select(self, intent: str | None) -> list[dict]:
        """Select documents using an existing intent label."""
        if not intent:
            return []

        return [
            doc
            for doc in self.documents
            if intent in doc.get("related_intents", [])
        ]

    def get_context(self, intent: str | None) -> str:
        """Return factual text for the response model."""
        documents = self.select(intent)

        return "\n\n".join(
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
            for doc in documents
        )