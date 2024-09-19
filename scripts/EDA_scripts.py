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
        logging.info("Cleaning data")
        # Handling missing values
        imputer = SimpleImputer(strategy='mean')
        df['CompetitionDistance'] = imputer.fit_transform(df[['CompetitionDistance']])
        
        # Outlier detection using Z-score for 'Sales' and 'Customers'
        df = df[(np.abs(stats.zscore(df[['Sales', 'Customers']])) < 3).all(axis=1)]
        
        return df
    def handle_catagorical_values(self, df,columns):
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
        - df: DataFrame with 'Date', 'IsHoliday', and 'Sales' columns.
        
        Returns:
        - None
        """
        # Default to 'After Holiday'
        df['HolidayStatus'] = 'After Holiday'
        
        # Assign 'During Holiday'
        df.loc[df['IsHoliday'] == 1, 'HolidayStatus'] = 'During Holiday'
        
        # Assign 'Before Holiday' by shifting 'IsHoliday'
        df['IsNextDayHoliday'] = df['IsHoliday'].shift(-1).fillna(0).astype(int)
        df.loc[df['IsNextDayHoliday'] == 1, 'HolidayStatus'] = 'Before Holiday'
        
        # Group by 'HolidayStatus' and calculate average sales
        sales_by_period = df.groupby('HolidayStatus')['Sales'].mean().reset_index()

        # Plotting the sales behavior before, during, and after holidays
        plt.figure(figsize=(10, 6))
        sns.barplot(x='HolidayStatus', y='Sales', data=sales_by_period, palette='Set2')
        plt.title('Average Sales Before, During, and After Holidays')
        plt.ylabel('Average Sales')
        plt.xlabel('Period')
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
        plt.title('Sales Before, During, and After Holidays')
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
    
    def add_holiday_season(slef, df):
        """
        Adds columns for Christmas and Easter seasons to indicate if the date falls within these holiday periods.
        
        Parameters:
        - df: Pandas DataFrame with a 'Date' column in datetime format.
        
        Returns:
        - df: DataFrame with additional 'IsChristmas' and 'IsEaster' columns
        """
        logging.info("Adding holiday season columns...")
        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Christmas season: Dec 20 - Dec 31
        df['IsChristmasSeason'] = ((df['Date'].dt.month == 12) & (df['Date'].dt.day >= 20)) | (df['Date'].dt.month == 1)
        
        # Easter season: Calculate Easter and create a range for Easter season
        easter_dates = pd.to_datetime([easter.easter(year) for year in df['Date'].dt.year.unique()])
        
        # Easter season could be considered a few days before and after Easter Sunday
        df['IsEasterSeason'] = df['Date'].isin(easter_dates) | df['Date'].isin(easter_dates + pd.Timedelta(days=1)) | \
                            df['Date'].isin(easter_dates - pd.Timedelta(days=1))
        
        return df
    def plot_seasonal_sales(self, df, season_col, title):
        """
        Plots average sales during and outside a given holiday season.
        
        Parameters:
        - df: Pandas DataFrame with 'Sales' and a seasonal indicator column (0 or 1).
        - season_col: The name of the column indicating the season (e.g., 'IsChristmasSeason' or 'IsEasterSeason').
        - title: The title of the plot.
        """
        logging.info("Plotting seasonal sales effects...")
        # Calculate the average sales for each season
        plt.figure(figsize=(12, 6))
        seasonal_effect = df.groupby(season_col)['Sales'].mean()
        seasonal_effect.plot(kind='bar', color=['skyblue', 'green'])
        plt.title(title)
        plt.xlabel(f'{season_col}')
        plt.ylabel('Average Sales')
        plt.xticks(ticks=[0, 1], labels=['Non-Holiday', 'Holiday Season'])
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