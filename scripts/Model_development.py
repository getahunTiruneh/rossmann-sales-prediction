import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

class ModelBuilder:
    def __init__(self, data):
        self.data = data
        self.model = None

    def split_data(self, target_column):
        """
        Splits the dataset into features (X) and target (y).
        Args:
            target_column (str): The name of the target column to predict.
        
        Returns:
            X (pd.DataFrame): The feature variables.
            y (pd.Series): The target variable.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return X, y

    def build_random_forest(self):
        """
        Builds, trains, and evaluates a Random Forest Regressor.

        Splits the data, applies scaling, and fits the model in a pipeline. 
        Returns the trained model and evaluation metrics: MSE, RMSE, MAE, and R2.
        
        Returns:
            pipeline (Pipeline): The trained model pipeline.
            mse (float): Mean Squared Error.
            rmse (float): Root Mean Squared Error.
            mae (float): Mean Absolute Error.
            r2 (float): R-squared score.
        """
        X, y = self.split_data(target_column='Sales')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        
         # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-Squared (R2): {r2}')
        return pipeline, mse, rmse, mae, r2

    def serialize_model(self):
        """Serialize the model to a file with a timestamp in the filename."""
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        filename = f"random_forest_model-{timestamp}.pkl"

        # Serialize the model
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved as {filename}")
        return filename

    def feature_importance(self):
        """
        Extracts and returns feature importance from the Random Forest model.

        Returns:
            pd.DataFrame: Sorted feature importance or None if unavailable.
        """
        if hasattr(self.model.named_steps['rf'], 'feature_importances_'):
            importances = self.model.named_steps['rf'].feature_importances_
            features = self.preprocessor.train_df.drop(columns=['Sales']).columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            return importance_df.sort_values(by='Importance', ascending=False)
        else:
            print("No feature importances available.")
            return None