# openrouter-model-json

## Information on how the price levels are structured.
free: output_price_per_1m == 0
budget: 0 < output_price_per_1m <= 2.50
standard: 2.50 < output_price_per_1m <= 5.00
premium: output_price_per_1m > 5.00
