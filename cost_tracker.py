
import os


def estimate_tokens(text):
    """
    Rough token estimate for mock mode.
    This is not an exact tokenizer.
    """
    if not text:
        return 0

    return max(1, round(len(text.split()) * 1.3))


def calculate_cost(input_tokens, output_tokens):
    """
    Calculate estimated USD cost using configurable
    prices per one million tokens.
    """
    input_price = float(
        os.getenv("INPUT_PRICE_PER_MILLION", "0")
    )
    output_price = float(
        os.getenv("OUTPUT_PRICE_PER_MILLION", "0")
    )

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price

    return input_cost + output_cost


def format_currency(amount):
    return f"${amount:.6f}"
