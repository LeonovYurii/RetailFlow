from collections import defaultdict

import numpy as np

from agent1.routing.action_types import (
    ActionCandidate,
    ActionType,
    RoutingDecision,
)


class IntentRouter:
    """
    Aggregate intent probabilities into ranked Agent1 action candidates.

    The router does not decide whether clarification is required.
    It only produces routing evidence for downstream analysis.
    """

    def __init__(
        self,
        intent_to_action,
        min_action_score=0.40,
        min_action_margin=0.10,
        top_k=5,
    ):
        if not 0.0 <= min_action_score <= 1.0:
            raise ValueError(
                "min_action_score must be between 0 and 1."
            )

        if not 0.0 <= min_action_margin <= 1.0:
            raise ValueError(
                "min_action_margin must be between 0 and 1."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        self.intent_to_action = intent_to_action
        self.min_action_score = min_action_score
        self.min_action_margin = min_action_margin
        self.top_k = top_k

    def route(
        self,
        probabilities,
        classes,
    ):
        """
        Convert an intent-probability distribution into ranked action candidates.

        Steps:
        1. Select top-k intent predictions.
        2. Map each intent to an action.
        3. Aggregate probabilities by action.
        4. Rank action candidates.
        5. Return routing evidence and confidence information.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        classes = np.asarray(classes)

        # -----------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------

        if probabilities.ndim != 1:
            raise ValueError(
                "probabilities must be one-dimensional."
            )

        if classes.ndim != 1:
            raise ValueError(
                "classes must be one-dimensional."
            )

        if len(probabilities) != len(classes):
            raise ValueError(
                "probabilities and classes must have the same length."
            )

        if len(probabilities) == 0:
            raise ValueError(
                "probabilities cannot be empty."
            )

        # -----------------------------------------------------
        # Select top-k intent predictions
        # -----------------------------------------------------

        k = min(
            self.top_k,
            len(probabilities),
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:k]

        # -----------------------------------------------------
        # Aggregate probabilities by action
        # -----------------------------------------------------

        action_scores = defaultdict(float)

        action_intents = defaultdict(list)

        for idx in top_indices:
            intent = str(classes[idx])

            intent_score = float(
                probabilities[idx]
            )

            action = self.intent_to_action.get(
                intent,
                ActionType.GENERAL_RESPONSE,
            )

            action_scores[action] += intent_score

            action_intents[action].append(
                (
                    intent,
                    intent_score,
                )
            )

        # -----------------------------------------------------
        # Rank actions
        # -----------------------------------------------------

        ranked_actions = sorted(
            action_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        candidates = [
            ActionCandidate(
                action=action,
                score=float(score),
            )
            for action, score in ranked_actions
        ]

        best_action, best_action_score = (
            ranked_actions[0]
        )

        if len(ranked_actions) > 1:
            second_action_score = float(
                ranked_actions[1][1]
            )
        else:
            second_action_score = 0.0

        action_margin = (
            float(best_action_score)
            - second_action_score
        )

        # -----------------------------------------------------
        # Representative intent of the winning action
        # -----------------------------------------------------

        winning_intent, winning_intent_score = max(
            action_intents[best_action],
            key=lambda item: item[1],
        )

        # -----------------------------------------------------
        # Confidence diagnostics
        # -----------------------------------------------------

        low_score = (
            best_action_score
            < self.min_action_score
        )

        low_margin = (
            action_margin
            < self.min_action_margin
        )

        low_confidence = (
            low_score
            or low_margin
        )

        if low_score and low_margin:
            reason = (
                "Top action score is below threshold "
                "and top action candidates are close."
            )

        elif low_score:
            reason = (
                "Top action score is below threshold."
            )

        elif low_margin:
            reason = (
                "Top action candidates are close."
            )

        else:
            reason = (
                "Routing confidence is sufficient."
            )

        # -----------------------------------------------------
        # Return routing evidence
        # -----------------------------------------------------

        return RoutingDecision(
            intent=winning_intent,
            intent_score=float(
                winning_intent_score
            ),
            action=best_action,
            action_score=float(
                best_action_score
            ),
            action_margin=float(
                action_margin
            ),
            candidates=candidates,
            low_confidence=low_confidence,
            reason=reason,
        )