import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import stats
import holidays
from dateutil import easter
import logging

class CustomerBehaviorAnalyzer:
    def __init__(self, data):
        self.data = data
    # Setup logging configuration
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    def merge_data(self, train, test, store):
        """
        Merges the train and test data with the store data on the store column
        Args:
            train (pandas.DataFrame): Train data
            test (pandas.DataFrame): Test data
            store (pandas.DataFrame): Store data
            
        Returns:
            tuple: (merged_train, merged_test)
        """
        logging.info("Merging train and test data with store data")
        merged_train = pd.merge(train, store, how='left', on='Store')
        merged_test = pd.merge(test, store,how='left', on='Store')
        return merged_train, merged_test
    
    # Data cleaning function
    def clean_data(self, df):
        """
        Cleans the input DataFrame by handling missing values and removing outliers.
        Args:
            df (pandas.DataFrame): Input DataFrame
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        logging.info("Cleaning data")
        # Handling missing values
        imputer = SimpleImputer(strategy='mean')
        df['CompetitionDistance'] = imputer.fit_transform(df[['CompetitionDistance']])
        
        # Outlier detection using Z-score for 'Sales' and 'Customers'
        df = df[(np.abs(stats.zscore(df[['Sales', 'Customers']])) < 3).all(axis=1)]
        
        return df
    def handle_catagorical_values(self, df,columns):
        """
        Handles categorical values in the dataset by converting them to numerical using one-hot encoding.
        Args:
            df (pandas.DataFrame): Input DataFrame
            columns (list): List of column names to be encoded
        Returns:
            pandas.DataFrame: DataFrame with categorical values encoded
        """
        logging.info("Handling categorical values")
        # Convert categorical variables to numerical using one-hot encoding
        df = pd.get_dummies(df, columns)
        return df
    
    # Plotting the distribution of promotions in training and test sets
    def plot_promo_distribution(self, merge_train_df, merge_test_df):
        """
        Plots the distribution of promotions in the training and test sets.
        Args:
            merge_train_df (pandas.DataFrame): Merged train data
            merge_test_df (pandas.DataFrame): Merged test data
        """
        logging.info("Plotting promotion distribution")
        # Plotting the distribution of promotions
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.histplot(merge_train_df['Promo'], kde=False, ax=ax[0])
        ax[0].set_title('Promo Distribution - Training Set')
        
        sns.histplot(merge_test_df['Promo'], kde=False, ax=ax[1], color='green')
        ax[1].set_title('Promo Distribution - Test Set')
        
        plt.show()
    
    def add_holiday_columns(self, df):
        """
        Adds a column 'IsHoliday' to indicate if the date is a holiday.
        Parameters:
        - df: Pandas DataFrame with a 'Date' column in datetime format.
        Returns:
        - df: DataFrame with an additional 'IsHoliday' column (1 if it's a holiday, 0 otherwise)
        """
        logging.info("Adding holiday columns...")
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Get unique years from the 'Date' column
        years = df['Date'].dt.year.unique()

        # Define ET holidays for the years present in the dataset for Ethiopia
        etiopian_holidays = holidays.ET(years=years)

        # Add 'IsHoliday' column based on whether the date is in the holiday list
        df['IsHoliday'] = df['Date'].isin(etiopian_holidays).astype(int)

        return df
    def plot_holiday_effects(self, df):
        """
        Adds a 'HolidayStatus' column to indicate before, during, and after holidays,
        and plots the average sales behavior for each period.
        
        Parameters:
        - df: DataFrame with 'Date', 'StateHoliday', and 'Sales' columns.
        
        Returns:
        None 
        """
        logging.info("Plotting holiday effects...")
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Default to 'Regular Day'
        df['HolidayStatus'] = 'Regular Day'
        
        # Assign 'During Holiday' where StateHoliday is not 0 (assuming 0 = no holiday)
        df.loc[df['StateHoliday'] != '0', 'HolidayStatus'] = 'During Holiday'
        
        # Assign 'Before Holiday' using shift() to identify the day before a holiday
        df['IsNextDayHoliday'] = df['StateHoliday'].shift(-1).fillna('0')
        df.loc[df['IsNextDayHoliday'] != '0', 'HolidayStatus'] = 'Before Holiday'
        
        # Assign 'After Holiday' using shift() to identify the day after a holiday
        df['IsPrevDayHoliday'] = df['StateHoliday'].shift(1).fillna('0')
        df.loc[df['IsPrevDayHoliday'] != '0', 'HolidayStatus'] = 'After Holiday'
        
        # Group by 'HolidayStatus' and calculate average sales
        sales_by_period = df.groupby('HolidayStatus')['Sales'].mean().reset_index()

        # Plotting the sales behavior before, during, and after holidays
        plt.figure(figsize=(10, 6))
        sns.barplot(x='HolidayStatus', y='Sales', data=sales_by_period, palette='Set2')
        plt.title('Average Sales Before, During, and After Holidays')
        plt.ylabel('Average Sales')
        plt.xlabel('Holiday Period')
        plt.show()
  
    # Assuming 'StateHoliday' is a binary feature and 'Sales' is the target
    def plot_sales_holiday_behavior(self, df):
        """
        Plots the average sales before holidays, during holidays, and after holidays.

        Parameters:
        - df: DataFrame with 'StateHoliday' (binary) and 'Sales' columns
        """
        logging.info("Plotting sales effects due to holidays...")
        # plot sales before, during, and after holidays
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df['IsHoliday'])
        plt.title('Sales on Holidays and non-Holidays')
        plt.show()
    # Correlation analysis and scatter plot
    def correlation_sales_customers(self, df):
        """
        Calculates the correlation between the number of customers and sales,
        and creates a scatter plot to visualize the relationship.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.

        Returns:
            None
        """
        logging.info("Calculating correlation between sales and customers...")
        # Extract year from the 'Date' column
        df["Year"]=pd.DatetimeIndex(df['Date']).year
        # Calculate the correlation
        correlation = df['Sales'].corr(df['Customers'])
        print(f'Correlation between Sales and Customers: {correlation}')

        # Create a scatter plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Sales', y='Customers',hue='Year', data=df, alpha=0.8)
        plt.title('Sales vs. Number of Customers')
        plt.show()
        
    # Plotting sales with and without promotions
    def plot_sales_promo_effect(self, df):
        """
        Plots a boxplot to visualize the effect of promotions on sales.
        Args:
            df (pandas.DataFrame): The DataFrame containing the data.

        Returns:
            None
        """
        logging.info("Plotting sales effects due to promotions...")
        # Create a boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Promo', y='Sales', data=df)
        plt.title('Sales with and without Promotions')
        plt.show()
    
    # Function to analyze and plot the effect of state holidays on sales
    def plot_sales_by_state_holiday(self, df):
        """
        Plots the effect of state holidays on sales, with labeled holiday categories.
        - df: DataFrame containing sales and StateHoliday columns.
        Returns:
        None
        """
        logging.info("Plotting sales effects due to state holidays...")
        # Mapping the StateHoliday values to descriptive labels
        holiday_mapping = {
            '0': 'No Holiday',
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas Holiday'
        }
        
        # Apply the mapping to the StateHoliday column
        df['StateHoliday'] = df['StateHoliday'].map(holiday_mapping)

        # Group data by StateHoliday and calculate the mean sales for each category
        sales_by_holiday = df.groupby('StateHoliday')['Sales'].mean().reset_index()
        
        # Create a bar plot to visualize the sales by holiday type
        plt.figure(figsize=(8, 6))
        sns.barplot(x='StateHoliday', y='Sales', data=sales_by_holiday)
        plt.title('Average Sales During State Holidays')
        plt.xlabel('Holiday Type')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.show()
    # Scatter plot to check distance vs sales relationship
    def competitor_distance_sales(self, df):
        """
        Creates a scatter plot to analyze the relationship between competitor distance and sales.

        Parameters:
        - df: Pandas DataFrame with 'CompetitionDistance' and 'Sales' columns.

        Returns:
        None
        """
        logging.info("Plotting competitor distance vs sales...")
        # Create a scatter plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
        plt.title('Competitor Distance vs Sales')
        plt.show()
    # we have the columns 'Open' is a binary feature (1 if store is open, 0 if closed)
    def customer_opening_behavior(self, df):
        """
        Plots the customer traffic based on whether the store is open or closed.

        Parameters:
        - df: Pandas DataFrame with 'Open' and 'Customers' columns.

        Returns:
        None
        """
        logging.info("Plotting customer traffic based on store opening status...")
        # Line plot showing customer visits over time (dates), with a distinction between open and closed times
        plt.figure(figsize=(12, 6))

        # Plot customer traffic based on store opening (1 for open, 0 for closed)
        sns.lineplot(x='DayOfWeek', y='Customers', hue='Open', data=df, palette="viridis", style="Open", markers=True, linewidth=2.5)

        # Add labels and title
        plt.title('Customer Traffic Based on Store Opening and Closing')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Customers')
        plt.legend(title='Store Open (1 = Open, 0 = Closed)')
        plt.grid(True)
        plt.tight_layout()

        # Show plot
        plt.show()
    # Plotting sales based on assortment type
    def assortment_sales(self, df):
        """
        Creates a boxplot to analyze the effect of assortment type on sales.

        Parameters:
        - df: Pandas DataFrame with 'Assortment' and 'Sales' columns.

        Returns:
        None
        """
        logging.info("Plotting sales based on assortment type...")
        # Create a boxplot for assortment and sales 
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Assortment', y='Sales', data=df)
        plt.title('Sales by Assortment Type')
        plt.show()
    # Sample function to plot sales before and after competitor entry
    def plot_sales_vs_competitor(self, df):
        """
        Plots the sales before and after a competitor enters the store.

        Parameters:
        - df: Pandas DataFrame with 'Date', 'Sales', and 'CompetitionDistance' columns.

        Returns:
        None
        """
        logging.info("Plotting sales before and after competitor entry...")
        plt.figure(figsize=(12, 6))

        # Convert 'Date' to datetime for proper time series plotting
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the sales as a time series
        sns.lineplot(x='Date', y='Sales', data=df, label='Store Sales', marker='o')

        # Mark the point when a competitor enters (CompetitorDistance changes from NA to a number)
        competitor_entry_date = df.loc[df['CompetitionDistance'].notna(), 'Date'].min()
        
        if pd.notna(competitor_entry_date):
            plt.axvline(competitor_entry_date, color='red', linestyle='--', label='Competitor Entry')

        # Add labels and title
        plt.title('Impact of New Competitors on Store Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()
        
    def get_top_performing_stores(self, df, top_n=4):
        """
        Identify top-performing stores based on total sales.
        
        Parameters:
        - df: Pandas DataFrame containing 'Store' and 'Sales' columns.
        - top_n: Number of top stores to return. Default is 10.
        
        Returns:
        - top_stores_df: DataFrame of top-performing stores.
        """
        logging.info("Identifying top-performing stores...")
        # Aggregate sales by store
        store_sales = df.groupby('StoreType')['Sales'].sum().reset_index()
        
        # Sort stores by sales in descending order and select the top N stores
        top_stores_df = store_sales.sort_values(by='Sales', ascending=False).head(top_n)
        
        return top_stores_df
    def plot_top_performing_stores(self, top_stores_df):
        """
        Plot top-performing stores using a bar chart.
        
        Parameters:
        - top_stores_df: DataFrame of top-performing stores.
        """
        logging.info("Plotting top-performing stores...")
        # Create a bar chart for top performing stores
        plt.figure(figsize=(10, 6))
        sns.barplot(x='StoreType', y='Sales', data=top_stores_df, palette='viridis')
        plt.title('Top Performing Stores Types by Total Sales')
        plt.xlabel('Store Type')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.show()
    def analyze_promo_effectiveness(self, df):
        """
        Analyze the effectiveness of promotions by comparing sales with and without promos.
        Identify which stores benefit the most from promotions.
        
        Parameters:
        - df: DataFrame containing 'Store', 'Promo', 'Sales', and other relevant columns.
        
        Returns:
        - promo_analysis: DataFrame with store-level analysis of promo effectiveness.
        """
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Now, analyze promo effectiveness at the store level
        store_sales_by_promo = df.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack().fillna(0)

        # Calculate sales uplift during promotions (Promo = 1 vs Promo = 0)
        store_sales_by_promo['SalesUplift'] = store_sales_by_promo[1] - store_sales_by_promo[0]

        # Sort stores by the most sales uplift during promotions
        promo_analysis = store_sales_by_promo.sort_values(by='SalesUplift', ascending=False).reset_index()

        # Plot top 10 stores with the highest sales uplift during promos
        plt.figure(figsize=(10, 6))
        sns.barplot(x='SalesUplift', y='StoreType', data=promo_analysis.head(10), palette='coolwarm')
        plt.title('Stores Types with Highest Sales Uplift During Promotions')
        plt.xlabel('Sales Uplift')
        plt.ylabel('Store')
        plt.xticks(rotation=45, ha='right')
        plt.show()