import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yfinance as yf

# Function to download data and handle errors


def download_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data
    except Exception as e:
        print(f"Failed to get ticker '{ticker}' reason: {e}")
        sys.exit(1)

# Download historical data


ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
data = download_data(ticker, start_date, end_date)

# Calculate daily returns
data['Return'] = data['Adj Close'].pct_change()

# Create target variable: 1 if next day return is positive, 0 if negative
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)

# Drop NaN values
data.dropna(inplace=True)

# Create lagged return features
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Lag3'] = data['Return'].shift(3)
data['Lag4'] = data['Return'].shift(4)
data['Lag5'] = data['Return'].shift(5)
data.dropna(inplace=True)

# Define features and target
features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']
target = 'Target'

# Check if there's enough data for training
if len(data) == 0:
    print("No data available for training.")
    sys.exit(1)

# Split the data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy:.2f}')

# Make a prediction for the next day
latest_data = data[features].iloc[-1].values.reshape(1, -1)
predicted_direction = model.predict(latest_data)
print(f'Predicted direction: {"Up" if predicted_direction[0] == 1 else "Down"}')
