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
        logging.info(f"Processing datetime columns: {datetime_columns}")
        
        for col in datetime_columns:
            self.train_df[col] = pd.to_datetime(self.train_df[col], errors='coerce')

            # Extract features like year, month, day, weekday, etc.
            self.train_df[f'{col}_year'] = self.train_df[col].dt.year
            self.train_df[f'{col}_month'] = self.train_df[col].dt.month
            self.train_df[f'{col}_day'] = self.train_df[col].dt.day
            self.train_df[f'{col}_weekday'] = self.train_df[col].dt.weekday
            self.train_df[f'{col}_is_weekend'] = (self.train_df[col].dt.weekday >= 5).astype(int)

            # Drop the original datetime column
            self.train_df.drop(columns=[col], inplace=True)

        return self.train_df
    
    def merge_store_data(self):
        """Merge store information with train and test datasets."""
        logging.info("Merging store data with train and test datasets.")
        self.train_df = pd.merge(self.train_df, self.store_df, how='left', on='Store')
        self.test_df = pd.merge(self.test_df, self.store_df, how='left', on='Store')
        return self.train_df, self.test_df

    def add_date_features(self, train_df, test_df):
        """Extract date-related features such as weekdays, weekends, and holidays."""
        logging.info("Adding date features.")
        
        for df in [train_df, test_df]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Weekday'] = df['Date'].dt.weekday
            df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year

            # Use the stateHoliday feature directly to indicate holidays
            df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)

            # Drop StateHoliday column since it's no longer needed
            df.drop('StateHoliday', axis=1, inplace=True)

        return train_df, test_df

    def preprocess(self):
        """Run all preprocessing steps on both train and test datasets."""
        logging.info("Running preprocessing on both train and test datasets.")
        
        self.train_df, self.test_df = self.merge_store_data()

        # Add date features
        self.train_df, self.test_df = self.add_date_features(self.train_df, self.test_df)

        return self.train_df, self.test_df

    def check_missing_values(self, df):
        """Check and log the percentage of missing values in the DataFrame."""
        missing_fraction = df.isnull().mean()
        logging.info(f"Missing values per column:\n{missing_fraction}")
        return missing_fraction

    def handle_missing_data(self, train_df, test_df, num_strategy='mean', cat_strategy='mode', threshold=0.49):
        """Handle missing data by dropping columns and imputing based on strategies."""
        logging.info("Handling missing data")

        # Handle for both datasets
        for df in [train_df, test_df]:
            missing_fraction = self.check_missing_values(df)
            
            # Identify columns where the missing fraction exceeds the threshold
            columns_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
            logging.info(f"Columns to drop (missing fraction > {threshold}): {columns_to_drop}")
            
            # Drop the identified columns
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                logging.info(f"Columns dropped: {columns_to_drop}")
            else:
                logging.info("No columns dropped due to missing values exceeding threshold.")

            # Separate numerical and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=[object]).columns

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

        return train_df, test_df

    def encode_categorical_values(self, train_df, test_df, method='onehot'):
        """Encode categorical variables using specified encoding method."""
        logging.info(f"Encoding categorical values using {method} method.")
        
        categorical_cols = train_df.select_dtypes(include=[object]).columns

        if method == 'onehot':
            # Fit and transform the train DataFrame
            train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
            
            # Transform the test DataFrame using the same columns as the train DataFrame
            test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
            
            # Align the columns of test_df with train_df, filling missing columns with 0
            # test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            
            # Fit and transform for each categorical column in train_df
            for col in categorical_cols:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                label_encoders[col] = le  # Save the encoder
                
                # Transform the test DataFrame using the same label encoder
                test_df[col] = le.transform(test_df[col].fillna('Unknown'))  # Handle unseen categories
        # Convert boolean values to integers (1 and 0)
        bool_cols = train_df.select_dtypes(include=[bool]).columns
        bool_cols_test = test_df.select_dtypes(include=[bool]).columns
        train_df[bool_cols] = train_df[bool_cols].astype(int)
        test_df[bool_cols_test] = test_df[bool_cols_test].astype(int)

        return train_df, test_df

    def detect_outliers(self, train_df, test_df, method='zscore', threshold=3):
        """Detect and handle outliers in the numerical columns."""
        logging.info("Detecting and handling outliers.")
        
        for df in [train_df, test_df]:
            # Select only numeric columns for outlier detection
            numeric_cols = df.select_dtypes(include=[np.number])
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(numeric_cols))
                df = df[(z_scores < threshold).all(axis=1)]
            
            elif method == 'iqr':
                Q1 = numeric_cols.quantile(0.25)
                Q3 = numeric_cols.quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return train_df, test_df

    def run_pipeline(self, missing_num_strategy='mean', missing_cat_strategy='mode', outlier_method='zscore'):
        """Run the entire preprocessing pipeline."""
        logging.info("Running the complete preprocessing pipeline.")
        
        # Preprocess datasets
        self.train_df, self.test_df = self.preprocess()
        # Handle missing data
        self.train_df, self.test_df = self.handle_missing_data(self.train_df, self.test_df, num_strategy=missing_num_strategy, cat_strategy=missing_cat_strategy)
        # Encode categorical values
        self.train_df, self.test_df = self.encode_categorical_values(self.train_df, self.test_df)
        
        # Detect and handle outliers
        self.train_df, self.test_df = self.detect_outliers(self.train_df, self.test_df, method=outlier_method)
        
        # Drop the Date column if it exists
        for df in [self.train_df, self.test_df]:
            if 'Date' in df.columns:
                df.drop(columns=['Date'], inplace=True)
            if 'Customers' in df.columns:
                df.drop(columns=['Customers'], inplace=True)
                
        return self.train_df, self.test_df
