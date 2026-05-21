import yfinance as yf
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

pltr = yf.Ticker("AAPL")

print(json.dumps(pltr.info, indent=4)) 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(pltr.quarterly_income_stmt.to_string())

# Extract data into a DataFrame
hist = pltr.history(period='6mo')
print(hist.to_string())

# Plot 1: Closing Price
plt.figure(figsize=(12, 6))
sns.lineplot(data=hist, x=hist.index, y='Close', label='Closing Price', color='blue')
plt.title('Stock Closing Price (6 Months)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Trading Volume
plt.figure(figsize=(12, 4))
plt.bar(hist.index, hist['Volume'], color='orange', alpha=0.7)
plt.title('Trading Volume (6 Months)')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(axis='y')
plt.show()

first_exp = pltr.options[0]
print(pltr.option_chain(first_exp).calls.head().to_string())