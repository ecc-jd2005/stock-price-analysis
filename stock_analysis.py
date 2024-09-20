import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ticker = input("Enter the stock ticker symbol (e.g., HOG for Harley-Davidson): ").strip().upper()
data = yf.download(ticker, start="2010-01-01", end="2023-09-01")

if data.empty:
    print("No data found for the ticker symbol. Please check the symbol and try again.")
else:
    print(f"Analyzing the stock data for {ticker}.")
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label=f'{ticker} Closing Price', color='blue')
    plt.title(f'{ticker} Stock Price (2010-2023)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data['Future Price'] = data['Close'].shift(-1)
    data['Target'] = np.where(data['Future Price'] > data['Close'], 1, 0)
    data.dropna(inplace=True)

    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = [1 if x > 0.5 else 0 for x in predictions]

    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data[-100:], x='Date', y='Close', label='Actual Price', color='blue')
    plt.title('Predicted vs Actual Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
