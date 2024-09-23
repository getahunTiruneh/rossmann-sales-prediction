import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# LSTM Model Class
class LSTMModelBuilder:
    def __init__(self, df, n_lag=14):
        self.df = df
        self.n_lag = n_lag
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def check_stationarity(self):
        """Check whether the time series is stationary."""
        result = adfuller(self.df['Sales'])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] > 0.05:
            print("The data is non-stationary, differencing is required.")
            self.df['Sales_diff'] = self.df['Sales'].diff().dropna()
            self.df = self.df.dropna()
        else:
            print("The data is stationary.")
            self.df['Sales_diff'] = self.df['Sales']  # No need for differencing
    
    def plot_acf_pacf(self):
        """Plot Autocorrelation and Partial Autocorrelation."""
        plot_acf(self.df['Sales_diff'], lags=50)
        plot_pacf(self.df['Sales_diff'], lags=50)
        plt.show()

    def create_supervised_data(self):
        """Transform the time series data into supervised learning data."""
        X, y = [], []
        data = self.df['Sales_diff'].values
        for i in range(len(data) - self.n_lag):
            X.append(data[i:i + self.n_lag])
            y.append(data[i + self.n_lag])
        return np.array(X), np.array(y)
    
    def scale_data(self, X, y):
        """Scale the data using MinMaxScaler."""
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled
    
    def split_data(self, X, y, train_size=0.8):
        """Split the data into train and test sets."""
        train_size = int(len(X) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape the data to be 3D for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self):
        """Build and compile the LSTM model."""
        model = tf.keras.Sequential()

        # LSTM layers
        model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(self.n_lag, 1)))
        model.add(tf.keras.layers.LSTM(50, activation='relu'))

        # Dense output layer
        model.add(tf.keras.layers.Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the LSTM model."""
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        return model

    def plot_predictions(self, model, X_test, y_test):
        """Plot the actual vs predicted sales."""
        y_pred = model.predict(X_test)

        # Inverse transform to original scale
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        y_test_inv = self.scaler.inverse_transform(y_test)

        # Plot the results
        plt.figure(figsize=(10,6))
        plt.plot(y_test_inv, label='Actual Sales')
        plt.plot(y_pred_inv, label='Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        plt.legend()
        plt.show()
