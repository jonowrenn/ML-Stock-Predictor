# ML-Stock-Predictor
ML Stock Predictor made with PyTorch.

This project uses an artificial recurrent neural network (Long Short Term Memory, LSTM) to predict the closing stock price of a corporation (Apple Inc.) based on historical stock price data. The model is trained on the past 60 days of stock prices to predict future stock prices.

Features
Download historical stock price data using the yfinance library.
Preprocess and scale data for training using MinMaxScaler.
Build and train an LSTM neural network for time-series forecasting.
Visualize actual vs. predicted stock prices using Matplotlib.
Predict the closing stock price for the next day.
Technologies Used
Python: Core programming language.
TensorFlow/Keras: For building and training the LSTM model.
Pandas: Data manipulation and analysis.
NumPy: Numerical computations.
yfinance: Fetching historical stock data.
Matplotlib: Data visualization.

Data Source
Historical stock data is fetched using the Yahoo Finance API via the yfinance library.
Project Workflow
Data Download and Preprocessing:

Fetch historical data (2012–2024) for Apple Inc. (AAPL).
Normalize the closing prices for better performance during training.
Model Building and Training:

Train the LSTM model on 80% of the data.
Use the remaining 20% for validation.
Predictions and Visualization:

Compare the predicted stock prices with the actual stock prices on the validation set.
Visualize the results using Matplotlib.
Future Predictions:

Predict the closing stock price for the next trading day.
