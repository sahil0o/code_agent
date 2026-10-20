def apply_discount(price, percent):
    """Return the price after taking `percent` percent off (percent is 0-100)."""
    return price * percent / 100


def format_price(amount):
    """Format a number as a rupee string, e.g. 1234.5 -> '₹1,234.50'."""
    return f"₹{amount:,.2f}"