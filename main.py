import yfinance as yf
import pandas as pd
import json

pltr = yf.Ticker("AAPL")

print(json.dumps(pltr.info, indent=4)) 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(pltr.quarterly_income_stmt.to_string())

print(pltr.history(period='6mo').to_string())

first_exp = pltr.options[0]
print(pltr.option_chain(first_exp).calls.head().to_string())