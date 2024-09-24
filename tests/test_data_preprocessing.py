import unittest
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer

class DataProcessingTest(unittest.TestCase):

    def setUp(self):
        """
        Sets up sample data for testing
        """
        # Sample data for train, test, and store
        self.train = pd.DataFrame({
            'Store': [1, 2, 3],
            'Sales': [100, 150, 200],
            'Customers': [50, 60, 70],
            'CompetitionDistance': [1000, np.nan, 2000]
        })
        
        self.test = pd.DataFrame({
            'Store': [1, 2, 3],
            'Sales': [120, 140, 210],
            'Customers': [55, 65, 75],
            'CompetitionDistance': [1050, np.nan, 2500]
        })

        self.store = pd.DataFrame({
            'Store': [1, 2, 3],
            'StoreType': ['A', 'B', 'C'],
            'Assortment': ['Basic', 'Extended', 'Extra']
        })
        
        self.processor = DataProcessing()

    def test_merge_data(self):
        """
        Test merging of train and test data with store data
        """
        merged_train, merged_test = self.processor.merge_data(self.train, self.test, self.store)
        
        # Check if the merge has the expected number of rows
        self.assertEqual(merged_train.shape[0], self.train.shape[0])
        self.assertEqual(merged_test.shape[0], self.test.shape[0])
        
        # Check if store-related columns are added
        self.assertIn('StoreType', merged_train.columns)
        self.assertIn('Assortment', merged_test.columns)

    def test_clean_data(self):
        """
        Test cleaning of data: handling missing values and outliers
        """
        cleaned_train = self.processor.clean_data(self.train)
        
        # Check if missing values in 'CompetitionDistance' have been filled
        self.assertFalse(cleaned_train['CompetitionDistance'].isnull().any())
        
        # Check if outliers in 'Sales' and 'Customers' have been removed
        z_scores = np.abs(stats.zscore(cleaned_train[['Sales', 'Customers']]))
        self.assertTrue((z_scores < 3).all().all())

    def test_handle_categorical_values(self):
        """
        Test handling of categorical values by converting them to dummy variables
        """
        columns_to_encode = ['StoreType', 'Assortment']
        df = pd.merge(self.train, self.store, on='Store')
        
        df_encoded = self.processor.handle_catagorical_values(df, columns_to_encode)
        
        # Check if original categorical columns are removed
        for col in columns_to_encode:
            self.assertNotIn(col, df_encoded.columns)
        
        # Check if dummy columns are added
        self.assertIn('StoreType_A', df_encoded.columns)
        self.assertIn('Assortment_Basic', df_encoded.columns)


# Class with data processing functions
class DataProcessing:
    
    def merge_data(self, train, test, store):
        """
        Merges the train and test data with the store data on the store column
        """
        merged_train = pd.merge(train, store, how='left', on='Store')
        merged_test = pd.merge(test, store, how='left', on='Store')
        return merged_train, merged_test

    def clean_data(self, df):
        """
        Cleans data by handling missing values and detecting/removing outliers
        """
        imputer = SimpleImputer(strategy='mean')
        df['CompetitionDistance'] = imputer.fit_transform(df[['CompetitionDistance']])
        df = df[(np.abs(stats.zscore(df[['Sales', 'Customers']])) < 3).all(axis=1)]
        return df
    
    def handle_catagorical_values(self, df, columns):
        """
        Handles categorical values by converting them to dummy variables
        """
        df = pd.get_dummies(df, columns)
        return df

if __name__ == '__main__':
    unittest.main()
