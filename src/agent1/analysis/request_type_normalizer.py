from collections.abc import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from agent1.analysis.action_plan import (
    REQUEST_TYPE_DESCRIPTIONS,
    RequestType,
)


class RequestTypeNormalizer:

    def __init__(
        self,
        encode: Callable,
        threshold: float = 0.60,
    ):
        self.encode = encode
        self.threshold = threshold

        self.request_types = list(
            REQUEST_TYPE_DESCRIPTIONS.keys()
        )

        self.descriptions = [
            REQUEST_TYPE_DESCRIPTIONS[request_type]
            for request_type in self.request_types
        ]

        self.description_embeddings = None

    def _get_description_embeddings(self):
        if self.description_embeddings is None:
            self.description_embeddings = self.encode(
                self.descriptions
            )

        return self.description_embeddings

    def _semantic_match(
        self,
        value: str,
    ) -> tuple[RequestType | None, float]:

        value_embedding = self.encode([
            f"The request type is {value}."
        ])

        description_embeddings = (
            self._get_description_embeddings()
        )

        similarities = cosine_similarity(
            value_embedding,
            description_embeddings,
        )[0]

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score < self.threshold:
            return None, best_score

        return self.request_types[best_index], best_score

    def _structurally_consistent(
        self,
        request_type: RequestType,
        data: dict,
    ) -> bool:

        if request_type == RequestType.CLARIFICATION:
            return (
                data.get("needs_clarification") is True
                and bool(data.get("clarification_question"))
                and data.get("action") is None
            )

        if request_type == RequestType.SINGLE_ACTION:
            return (
                data.get("action") is not None
                and data.get("needs_clarification") is not True
            )

        if request_type == RequestType.MULTI_ACTION:
            return (
                len(data.get("candidate_actions", [])) >= 2
                and data.get("needs_clarification") is not True
            )

        if request_type == RequestType.GENERAL_RESPONSE:
            return (
                data.get("action") is None
                and not data.get("candidate_actions")
                and data.get("needs_clarification") is not True
            )

        return False

    def normalize(
        self,
        data: dict,
    ) -> dict:

        value = data.get("request_type")

        # Already valid -- do nothing.
        try:
            RequestType(value)
            return data
        except (ValueError, TypeError):
            pass

        matched_type, score = self._semantic_match(
            str(value)
        )

        if matched_type is None:
            return data

        if not self._structurally_consistent(
            matched_type,
            data,
        ):
            return data

        data["request_type"] = matched_type.value

        return data