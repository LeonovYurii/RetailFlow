"""Small fixed inventory fixture; never represents live inventory."""
from agent1.tools.tool_result import ToolResult

class StockService:
    INVENTORY = {("backpack", "blue"): 5, ("backpack", "red"): 0}

    def check_stock(self, product=None, color=None):
        if not all(isinstance(v, str) and v.strip() for v in (product, color)):
            return ToolResult(success=False, action="availability",
                              message="product and color are required strings.")
        product, color = product.strip().lower(), color.strip().lower()
        data = {"product": product, "color": color, "source": "mock_inventory"}
        quantity = self.INVENTORY.get((product, color))
        if quantity is None:
            return ToolResult(success=False, action="availability", data=data,
                              message=f"Mock inventory has no record for {color} {product}; availability is unknown.")
        data.update(quantity=quantity, in_stock=quantity > 0)
        return ToolResult(success=True, action="availability", data=data,
                          message=f"Mock inventory: {color} {product}: {quantity} available.")
