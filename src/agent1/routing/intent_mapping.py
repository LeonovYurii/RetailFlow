from agent1.routing.action_types import ActionType


INTENT_TO_ACTION = {
    # ACCOUNT
    "change_account": ActionType.ACCOUNT_SERVICE,
    "close_account": ActionType.ACCOUNT_SERVICE,
    "open_account": ActionType.ACCOUNT_SERVICE,
    "order_history": ActionType.ACCOUNT_SERVICE,
    "recover_password": ActionType.AUTH_SUPPORT,

    # APP / WEBSITE
    "technical_issue": ActionType.APP_SUPPORT,
    "use_app": ActionType.APP_SUPPORT,

    # CART
    "add_product": ActionType.CART_SERVICE,
    "remove_product": ActionType.CART_SERVICE,

    # CONTACT
    "customer_service": ActionType.HUMAN_SUPPORT,
    "human_agent": ActionType.HUMAN_SUPPORT,

    # DELIVERY
    "damaged_delivery": ActionType.DELIVERY_SERVICE,
    "delivery_issue": ActionType.DELIVERY_SERVICE,
    "delivery_time": ActionType.DELIVERY_SERVICE,
    "missing_item": ActionType.DELIVERY_SERVICE,
    "shipping_costs": ActionType.DELIVERY_SERVICE,
    "track_delivery": ActionType.DELIVERY_SERVICE,
    "wrong_item": ActionType.DELIVERY_SERVICE,

    # FEEDBACK
    "submit_feedback": ActionType.FEEDBACK_SERVICE,
    "submit_product_feedback": ActionType.FEEDBACK_SERVICE,
    "submit_product_idea": ActionType.FEEDBACK_SERVICE,

    # ORDER
    "cancel_order": ActionType.ORDER_SERVICE,
    "change_order": ActionType.ORDER_SERVICE,
    "request_invoice": ActionType.ORDER_SERVICE,
    "track_order": ActionType.ORDER_SERVICE,

    # PAYMENT
    "pay": ActionType.PAYMENT_SERVICE,
    "payment_issue": ActionType.PAYMENT_SERVICE,
    "payment_methods": ActionType.PAYMENT_SERVICE,

    # PRODUCT
    "availability": ActionType.PRODUCT_SEARCH,
    "availability_in_store": ActionType.PRODUCT_SEARCH,
    "availability_online": ActionType.PRODUCT_SEARCH,
    "product_information": ActionType.PRODUCT_SEARCH,

    "exchange_product": ActionType.PRODUCT_SERVICE,
    "exchange_product_in_store": ActionType.PRODUCT_SERVICE,
    "product_issue": ActionType.PRODUCT_SERVICE,

    # RETURNS
    "refund_policy": ActionType.RETURNS_SERVICE,
    "refund_status": ActionType.RETURNS_SERVICE,
    "request_refund": ActionType.RETURNS_SERVICE,
    "return_policy": ActionType.RETURNS_SERVICE,
    "return_product": ActionType.RETURNS_SERVICE,
    "return_product_in_store": ActionType.RETURNS_SERVICE,
    "return_product_online": ActionType.RETURNS_SERVICE,

    # SALES
    "sales_period": ActionType.SALES_INFO,

    # STORE
    "store_location": ActionType.STORE_INFO,
    "store_opening_hours": ActionType.STORE_INFO,

    # USER / PRIVACY
    "request_right_to_rectification": ActionType.PRIVACY_SERVICE,
}