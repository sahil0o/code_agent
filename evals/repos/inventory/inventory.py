from dataclasses import dataclass

from pricing import apply_discount


@dataclass
class Item:
    name: str
    price: float
    quantity: int


class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, price, quantity=1):
        """Add stock. If the item already exists, increase its quantity."""
        if name in self.items:
            self.items[name].quantity += quantity
        else:
            self.items[name] = Item(name, price, quantity)

    def remove_item(self, name, quantity=1):
        """Remove stock. Delete the item entirely when its quantity reaches 0."""
        item = self.items[name]
        item.quantity -= quantity
        if item.quantity <= 0:
            del self.items[name]

    def total_value(self):
        """Return the total value of all stock (price x quantity for every item)."""
        total = 0.0
        for item in self.items.values():
            total += item.price * item.quantity
        return total

    def low_stock(self, threshold=5):
        """Return names of items whose quantity is at or below `threshold`."""
        return [item.name for item in self.items.values() if item.quantity < threshold]

    def sale_price(self, name, percent):
        """Return the price of one unit of `name` after a `percent` discount."""
        return apply_discount(self.items[name].price, percent)