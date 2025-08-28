#Train a model to detect and filter spam emails based on email content.
#Use techniques like Naive Bayes or Support Vector Machines (SVM) for classification.

# Stock Price Predictor using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Download Stock Data
df = yf.download("AAPL", start="2015-01-01", end="2025-01-01")
print("Data sample:\n", df.head())

# 2. Prepare Data
df = df[['Close']]  # use only closing price
df['Prediction'] = df[['Close']].shift(-30)  # predict 30 days into the future
X = np.array(df.drop(['Prediction'], axis=1))[:-30]  # Features
y = np.array(df['Prediction'])[:-30]  # Labels

# 3. Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate Model
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# 6. Predict Next 30 Days
future = np.array(df.drop(['Prediction'], axis=1))[-30:]  # last 30 days
future_predictions = model.predict(future)

# 7. Plot Results
valid = df[X.shape[0]:]
valid['Predictions'] = future_predictions

plt.figure(figsize=(10,5))
plt.title("Stock Price Prediction (AAPL)")
plt.xlabel("Date")
plt.ylabel("Close Price USD")
plt.plot(df['Close'], label='Actual Price')
plt.plot(valid['Predictions'], label='Predicted Price', linestyle='--')
plt.legend()
plt.show()
