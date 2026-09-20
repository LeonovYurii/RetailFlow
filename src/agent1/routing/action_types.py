from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    ACCOUNT_SERVICE = "account_service"
    AUTH_SUPPORT = "auth_support"
    APP_SUPPORT = "app_support"
    CART_SERVICE = "cart_service"
    HUMAN_SUPPORT = "human_support"
    DELIVERY_SERVICE = "delivery_service"
    FEEDBACK_SERVICE = "feedback_service"
    ORDER_SERVICE = "order_service"
    PAYMENT_SERVICE = "payment_service"
    PRODUCT_SEARCH = "product_search"
    PRODUCT_SERVICE = "product_service"
    RETURNS_SERVICE = "returns_service"
    SALES_INFO = "sales_info"
    STORE_INFO = "store_info"
    PRIVACY_SERVICE = "privacy_service"

    GENERAL_RESPONSE = "general_response"


@dataclass
class ActionCandidate:
    action: ActionType
    score: float


@dataclass
class RoutingDecision:
    intent: str
    intent_score: float

    action: ActionType
    action_score: float
    action_margin: float

    candidates: list[ActionCandidate]

    low_confidence: bool
    reason: str