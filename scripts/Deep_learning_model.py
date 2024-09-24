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
        """
        Checks if the 'Sales' data is stationary using the ADF test. If non-stationary (p > 0.05), applies 
        differencing and saves the result in 'Sales_diff'. If stationary, no differencing is applied.
        """
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
        """
        Plots the ACF and PACF for the differenced 'Sales' data ('Sales_diff') with 50 lags.
        """
        plot_acf(self.df['Sales_diff'], lags=50)
        plot_pacf(self.df['Sales_diff'], lags=50)
        plt.show()

    def create_supervised_data(self):
        """
        Transforms the differenced 'Sales' data into supervised learning format 
        by creating input-output pairs based on the specified lag.
        """
        X, y = [], []
        data = self.df['Sales_diff'].values
        for i in range(len(data) - self.n_lag):
            X.append(data[i:i + self.n_lag])
            y.append(data[i + self.n_lag])
        return np.array(X), np.array(y)
    
    def scale_data(self, X, y):
        """
        Scales features (X) and target (y) using MinMaxScaler.

        Parameters:
            X (ndarray): Input features to be scaled.
            y (ndarray): Target variable to be scaled.

        Returns:
            tuple: Scaled features (X_scaled) and scaled target (y_scaled).
        """
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled
    
    def split_data(self, X, y, train_size=0.8):
        """
        Split the data into train and test sets.

        Parameters:
            X (ndarray): Input features.
            y (ndarray): Target variable.
            train_size (float): Proportion of data to be used for training (default is 0.8).
        Returns:
            tuple: X_train, X_test, y_train, y_test split into training and testing sets.
        """
        train_size = int(len(X) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape the data to be 3D for LSTM (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self):
        """
        Build and compile the LSTM model.

        Returns:
            tf.keras.Model: Compiled LSTM model ready for training.
        """
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
        """
        Train the LSTM model.

        Parameters:
            model (tf.keras.Model): The LSTM model to train.
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target data.
            X_test (np.ndarray): Test input data.
            y_test (np.ndarray): Test target data.
            epochs (int, optional): Number of epochs for training. Default is 50.
            batch_size (int, optional): Size of batches for training. Default is 32.

        Returns:
            tuple: The trained model and the training history.
        """
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        return model, history

    def plot_predictions(self, model, X_test, y_test):
        """
        Plot the actual vs predicted sales.

        Parameters:
            model (tf.keras.Model): The trained LSTM model.
            X_test (np.ndarray): Test input data.
            y_test (np.ndarray): Actual sales data for the test set.

        Returns:
            None: This function displays a plot of actual vs predicted sales.
        """
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
        
    def plot_loss(self, history):
        """
        Plot training and validation loss over epochs.

        Parameters:
            history (tf.keras.callbacks.History): The training history returned by the model's fit method.

        Returns:
            None: This function displays a plot of training and validation loss.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
