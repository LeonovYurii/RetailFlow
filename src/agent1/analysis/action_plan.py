from enum import Enum

from pydantic import BaseModel, Field, model_validator

from agent1.routing.action_types import ActionType


class RequestType(str, Enum):
    SINGLE_ACTION = "single_action"
    MULTI_ACTION = "multi_action"
    CLARIFICATION = "clarification"
    GENERAL_RESPONSE = "general_response"


REQUEST_TYPE_DESCRIPTIONS = {
    RequestType.SINGLE_ACTION:
        "The user has one clear actionable request.",

    RequestType.MULTI_ACTION:
        "The user has multiple clear actionable requests.",

    RequestType.CLARIFICATION:
        "The user's request is ambiguous or unclear and "
        "requires clarification.",

    RequestType.GENERAL_RESPONSE:
        "The user does not require an agent action and "
        "needs a conversational response.",
}


class ActionStep(BaseModel):
    intent: str = Field(min_length=1)
    action: ActionType
    arguments: dict[str, str | None] = Field(default_factory=dict)
    # Dependencies/conditions are represented but deliberately not executed yet.
    depends_on: list[int] = Field(default_factory=list)
    condition: str | None = None


class ActionPlan(BaseModel):
    request_type: RequestType

    intent: str | None = None
    action: ActionType | None = None

    steps: list[ActionStep] = Field(default_factory=list)

    candidate_intents: list[str] = Field(default_factory=list)
    candidate_actions: list[ActionType] = Field(default_factory=list)

    needs_clarification: bool = False
    clarification_question: str | None = None

    reason: str

    @model_validator(mode="after")
    def validate_plan(self):

        if self.request_type == RequestType.CLARIFICATION:
            if not self.needs_clarification:
                raise ValueError(
                    "CLARIFICATION request type requires "
                    "needs_clarification=True."
                )

            if not self.clarification_question:
                raise ValueError(
                    "A clarification question is required."
                )

        else:
            if self.needs_clarification:
                raise ValueError(
                    "needs_clarification may only be True "
                    "for CLARIFICATION."
                )

            self.clarification_question = None

        if self.request_type == RequestType.SINGLE_ACTION:
            if self.action is None:
                raise ValueError(
                    "SINGLE_ACTION requires an action."
                )

        if self.request_type == RequestType.MULTI_ACTION:
            if len(self.steps) < 2:
                raise ValueError("MULTI_ACTION requires at least two explicit steps.")
            if self.intent is not None or self.action is not None:
                raise ValueError("MULTI_ACTION uses steps, not top-level intent/action.")
            if self.candidate_actions or self.candidate_intents:
                raise ValueError("Candidate alternatives must not represent execution steps.")
        elif self.steps:
            raise ValueError("Only MULTI_ACTION may contain steps.")

        if self.request_type == RequestType.GENERAL_RESPONSE:
            if self.action is not None:
                raise ValueError(
                    "GENERAL_RESPONSE must not contain an action."
                )

        return self