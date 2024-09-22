import pandas as pd
import numpy as np
from scipy import stats
import logging

class DataPreprocessingPipeline:
    
    def __init__(self, train_df, test_df, store_df):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.store_df = store_df.copy()
    
    def process_datetime_columns(self, datetime_columns):
        """Convert datetime columns into numerical features."""
        for col in datetime_columns:
            self.train_df[col] = pd.to_datetime(self.train_df[col], errors='coerce')

            # Extract features like year, month, day, weekday, etc.
            self.train_df[f'{col}_year'] = self.train_df[col].dt.year
            self.train_df[f'{col}_month'] = self.train_df[col].dt.month
            self.train_df[f'{col}_day'] = self.train_df[col].dt.day
            self.train_df[f'{col}_weekday'] = self.train_df[col].dt.weekday
            self.train_df[f'{col}_is_weekend'] = self.train_df[col].dt.weekday >= 5

            # Drop the original datetime column
            self.train_df.drop(columns=[col], inplace=True)

        return self.train_df
    
    def merge_store_data(self):
        """Merge store information with train and test datasets."""
        self.train_df = pd.merge(self.train_df, self.store_df, how='left', on='Store')
        self.test_df = pd.merge(self.test_df, self.store_df, how='left', on='Store')
        return self.train_df, self.test_df

    def add_date_features(self, df):
        """Extract date-related features such as weekdays, weekends, and holidays using the stateHoliday feature."""
        df['Date'] = pd.to_datetime(df['Date'])
        df['Weekday'] = df['Date'].dt.weekday
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

        # Use the stateHoliday feature directly to indicate holidays
        df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)

        return df

    def preprocess(self):
        """Run all preprocessing steps on both train and test datasets."""
        self.train_df, self.test_df = self.merge_store_data()

        # Add date features, using stateHoliday for holiday information
        self.train_df = self.add_date_features(self.train_df)
        self.test_df = self.add_date_features(self.test_df)

        # Handle missing values
        self.train_df.fillna(0, inplace=True)
        self.test_df.fillna(0, inplace=True)

        return self.train_df, self.test_df

    def handle_missing_data(self, df, num_strategy='mean', cat_strategy='mode', threshold=0.5):
        """Handle missing data with given strategies for numerical and categorical columns."""
        logging.info("Handling missing data")
        
        # Separate numerical and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=[object]).columns
        
        # Drop columns with too many missing values
        missing_fraction = df.isnull().mean()
        df = df.loc[:, missing_fraction < threshold]
        
        # Impute missing values for numerical columns
        if num_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif num_strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Impute missing values for categorical columns
        for col in categorical_cols:
            if cat_strategy == 'mode':
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value.iloc[0], inplace=True)
            elif cat_strategy == 'unknown':
                df[col].fillna('Unknown', inplace=True)
        
        return df

    def encode_categorical_values(self, df, method='onehot'):
        """Encode categorical variables using specified encoding method."""
        logging.info("Encoding categorical values")
        
        categorical_cols = df.select_dtypes(include=[object]).columns
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le  # Save the encoder if you need to inverse transform later
                
        return df

    def detect_outliers(self, df, method='zscore', threshold=3):
        """Detect and handle outliers in the numerical columns."""
        logging.info("Detecting and handling outliers")
        
        # Select only numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=[np.number])
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_cols))
            df = df[~(z_scores > threshold).any(axis=1)]
        
        elif method == 'iqr':
            Q1 = numeric_cols.quantile(0.25)
            Q3 = numeric_cols.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (numeric_cols < (Q1 - threshold * IQR)) | (numeric_cols > (Q3 + threshold * IQR))
            df = df[~(outliers).any(axis=1)]
        
        return df

    def run_pipeline(self, missing_num_strategy='mean', missing_cat_strategy='mode', outlier_method='zscore'):
        """_summary_

        Args:
            missing_num_strategy (str, optional): _description_. Defaults to 'mean'.
            missing_cat_strategy (str, optional): _description_. Defaults to 'mode'.
            outlier_method (str, optional): _description_. Defaults to 'zscore'.

        Returns:
            _type_: _description_
        """
        self.train_df, self.test_df = self.preprocess()

        # Handle missing data
        self.train_df = self.handle_missing_data(self.train_df, num_strategy=missing_num_strategy, cat_strategy=missing_cat_strategy)
        self.test_df = self.handle_missing_data(self.test_df, num_strategy=missing_num_strategy, cat_strategy=missing_cat_strategy)
        
        # Encode categorical values
        self.train_df = self.encode_categorical_values(self.train_df)
        self.test_df = self.encode_categorical_values(self.test_df)
        
        # Detect and handle outliers
        self.train_df = self.detect_outliers(self.train_df, method=outlier_method)
        self.test_df = self.detect_outliers(self.test_df, method=outlier_method)
        
        # Drop the Date column if it exists
        if 'Date' in self.train_df.columns:
            self.train_df.drop(columns=['Date'], inplace=True)
        if 'Date' in self.test_df.columns:
            self.test_df.drop(columns=['Date'], inplace=True)
            
        return self.train_df, self.test_df
