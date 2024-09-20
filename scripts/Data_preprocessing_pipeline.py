import pandas as pd
import numpy as np
from scipy import stats
import logging

class DataPreprocessingPipeline:
    
    def __init__(self, df):
        self.df = df.copy()

    # Step 1: Handle Missing Data
    def handle_missing_data(self, num_strategy='mean', cat_strategy='mode', threshold=0.5):
        logging.info("Handling missing data")        
        # Separate numerical and categorical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=[object]).columns
        
        # Drop columns with too many missing values
        missing_fraction = self.df.isnull().mean()
        self.df = self.df.loc[:, missing_fraction < threshold]
        
        print("Number of rows after dropping columns with too many missing values:", self.df.shape[0])
        
        # Impute missing values for numerical columns
        if num_strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif num_strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Impute missing values for categorical columns
        for col in categorical_cols:
            if cat_strategy == 'mode':
                mode_value = self.df[col].mode()
                if not mode_value.empty:  # Check if mode is available
                    self.df[col].fillna(mode_value.iloc[0], inplace=True)
            elif cat_strategy == 'unknown':
                self.df[col].fillna('Unknown', inplace=True)
        
        print("Number of rows after handling missing values:", self.df.shape[0])
        return self

    # Step 2: Detect and Handle Outliers
    def detect_outliers(self, method='zscore', threshold=3):
        logging.info("Detecting and handling outliers")
        # Select only numeric columns for outlier detection
        numeric_cols = self.df.select_dtypes(include=[np.number])
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_cols))
            outliers = (z_scores > threshold)
            self.df = self.df[~(outliers).any(axis=1)]
        
        elif method == 'iqr':
            Q1 = numeric_cols.quantile(0.25)
            Q3 = numeric_cols.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (numeric_cols < (Q1 - threshold * IQR)) | (numeric_cols > (Q3 + threshold * IQR))
            self.df = self.df[~(outliers).any(axis=1)]
        
        print("Number of rows after detecting and handling outliers:", self.df.shape[0])
        return self

    # Step 3: Run the entire pipeline
    def run_pipeline(self, missing_num_strategy='mean', missing_cat_strategy='mode', outlier_method='zscore'):
        """
        Run the entire pipeline.
        """
        self.handle_missing_data(num_strategy=missing_num_strategy, cat_strategy=missing_cat_strategy)
        self.detect_outliers(method=outlier_method)
        return self.df