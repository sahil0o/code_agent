from inventory import Inventory
from pricing import apply_discount, format_price


def make_inventory():
    inv = Inventory()
    inv.add_item("pen", 10.0, 20)
    inv.add_item("notebook", 50.0, 5)
    inv.add_item("bag", 500.0, 2)
    return inv


def test_add_existing_item_increases_quantity():
    inv = make_inventory()
    inv.add_item("pen", 10.0, 5)
    assert inv.items["pen"].quantity == 25


def test_remove_item_deletes_when_empty():
    inv = make_inventory()
    inv.remove_item("bag", 2)
    assert "bag" not in inv.items


def test_total_value():
    inv = make_inventory()
    assert inv.total_value() == 1450.0


def test_low_stock_includes_items_at_threshold():
    inv = make_inventory()
    assert sorted(inv.low_stock(threshold=5)) == ["bag", "notebook"]


def test_apply_discount():
    assert apply_discount(200.0, 25) == 150.0


def test_sale_price():
    inv = make_inventory()
    assert inv.sale_price("bag", 10) == 450.0


def test_format_price():
    assert format_price(1234.5) == "₹1,234.50"