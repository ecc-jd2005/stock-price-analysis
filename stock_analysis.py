import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ticker = input("Enter the stock ticker symbol (e.g., HOG for Harley Davidson, LMT for Lockheed Martin): ")

data = yf.download(ticker, start="2010-01-01", end="2024-01-01")

data['Date'] = data.index
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Close'], label='Actual Price', color='blue')
plt.plot(data.iloc[len(X_train):]['Date'], y_pred, label='Predicted Price', color='red', linestyle='dashed')
plt.title(f'{ticker} Predicted vs Actual Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
