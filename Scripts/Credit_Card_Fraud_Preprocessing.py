import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

class CreditCardFraudPreprocessing:
    def __init__(self):
        # Use the correct path for the Data folder
        self.creditcard_df = pd.read_csv('../Data/creditcard.csv')
        self.fraud_df = pd.read_csv('../Data/Fraud_Data.csv')
        self.ip_country_df = pd.read_csv('../Data/IpAddress_to_Country.csv')
        # Ensure the output directory exists
        os.makedirs('../Image', exist_ok=True)

    def handle_missing_values(self, data_type=None, strategy='mean'):
        """Handle missing values in datasets."""
        if data_type == 'creditcard':
            imputer = SimpleImputer(strategy=strategy)
            self.creditcard_df[['Amount', 'Time']] = imputer.fit_transform(self.creditcard_df[['Amount', 'Time']])
            print(f"Missing values handled in creditcard_df using strategy: {strategy}")

        elif data_type == 'fraud':
            # Use mean/median for numeric columns and most_frequent for categorical/string columns
            numeric_cols = self.fraud_df.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = self.fraud_df.select_dtypes(include=['object', 'datetime64']).columns
            
            # Impute numeric columns with the specified strategy (mean/median)
            num_imputer = SimpleImputer(strategy=strategy)
            self.fraud_df[numeric_cols] = num_imputer.fit_transform(self.fraud_df[numeric_cols])
            
            # Impute categorical columns with the most frequent strategy
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.fraud_df[categorical_cols] = cat_imputer.fit_transform(self.fraud_df[categorical_cols])
            
            print(f"Missing values handled in fraud_df. Numeric columns imputed with {strategy}, categorical columns with 'most_frequent'.")
    def data_cleaning(self, data_type=None):
        """Remove duplicates and correct data types for the specified dataset."""
        
        if data_type == 'creditcard':
            # Remove duplicates
            creditcard_initial_size = len(self.creditcard_df)
            self.creditcard_df.drop_duplicates(inplace=True)
            print(f"Duplicates removed - Creditcard dataset: {creditcard_initial_size - len(self.creditcard_df)}")
            
            # Correct data types
            self.creditcard_df['Time'] = pd.to_numeric(self.creditcard_df['Time'], errors='coerce')
            self.creditcard_df['Amount'] = pd.to_numeric(self.creditcard_df['Amount'], errors='coerce')
            
        elif data_type == 'fraud':
            # Remove duplicates
            fraud_initial_size = len(self.fraud_df)
            self.fraud_df.drop_duplicates(inplace=True)
            print(f"Duplicates removed - Fraud dataset: {fraud_initial_size - len(self.fraud_df)}")
        
        else:
            print("Please provide a valid data_type ('creditcard' or 'fraud').")
    def eda(self, data_type=None):
        """Perform Univariate and Bivariate Analysis with visualizations based on the dataset."""
        
        if data_type == 'creditcard':
            print("Univariate Analysis for Credit Card Dataset:")
            print(self.creditcard_df.describe())
            
            # Univariate: Distribution of the 'Amount' column
            plt.figure(figsize=(10, 6))
            sns.histplot(self.creditcard_df['Amount'], bins=50, color='#5D3FD3', kde=True)
            plt.title('Transaction Amount Distribution', fontsize=18)
            plt.xlabel('Amount', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.savefig('../Image/transaction_amount_distribution.png')
            plt.show()

            # Bivariate: Fraud vs Non-Fraud Analysis (Class distribution)
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Class', data=self.creditcard_df, palette=['#5D3FD3', '#FFBF00'])
            plt.title('Class Distribution (Fraud vs Non-Fraud)', fontsize=18)
            plt.xlabel('Class', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.savefig('../Image/class_distribution.png')
            plt.show()
            
        elif data_type == 'fraud':
            print("Univariate Analysis for Fraud Dataset:")
            print(self.fraud_df.describe())

            # Add relevant EDA visualizations for the fraud dataset here if needed.
        
        else:
            print("Please provide a valid data_type ('creditcard' or 'fraud').")
    def convert_ip_to_int(self, ip_str):
        """Convert an IP address to an integer."""
        if isinstance(ip_str, str):  # Check if the input is a string
            octets = ip_str.split('.')
            return int(octets[0]) * (256 ** 3) + int(octets[1]) * (256 ** 2) + int(octets[2]) * 256 + int(octets[3])
        else:
            return None  # Return None or some default value for invalid inputs

    def feature_engineering(self):
        """Create time-based features and transaction velocity."""
        # Time-based features for fraud_df
        self.fraud_df['hour_of_day'] = pd.to_datetime(self.fraud_df['transaction_time']).dt.hour
        self.fraud_df['day_of_week'] = pd.to_datetime(self.fraud_df['transaction_time']).dt.dayofweek

        # Transaction frequency and velocity
        self.fraud_df['transaction_count'] = self.fraud_df.groupby('customer_id')['transaction_time'].transform('count')
        self.fraud_df['transaction_velocity'] = self.fraud_df['transaction_amount'] / self.fraud_df['transaction_count']

        print("Feature engineering completed. Added columns 'hour_of_day', 'day_of_week', 'transaction_count', and 'transaction_velocity'.")

        # Visualization of time-based features
        plt.figure(figsize=(10, 6))
        sns.histplot(self.fraud_df['hour_of_day'], bins=24, color='#FF6347', kde=True)
        plt.title('Transaction Frequency by Hour of Day', fontsize=18)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig('../Image/transaction_by_hour.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(self.fraud_df['day_of_week'], bins=7, color='#4CAF50', kde=True)
        plt.title('Transaction Frequency by Day of Week', fontsize=18)
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.savefig('../Image/transaction_by_day.png')
        plt.show()

    def normalization_scaling(self, data_type=None, method='standard'):
        """Normalize and scale numerical features based on the dataset."""
        
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        
        if data_type == 'creditcard':
            self.creditcard_df[['Amount', 'Time']] = scaler.fit_transform(self.creditcard_df[['Amount', 'Time']])
            print(f"Credit Card data scaled using {method} scaling.")
            print("First few rows of the scaled Credit Card data:")
            print(self.creditcard_df[['Amount', 'Time']].head())
        
        elif data_type == 'fraud':
            # Assuming there are numerical columns to scale in fraud_df
            # Adjust column names according to the fraud_df dataset
            self.fraud_df[['transaction_amount']] = scaler.fit_transform(self.fraud_df[['transaction_amount']])
            print(f"Fraud data scaled using {method} scaling.")
            print("First few rows of the scaled Fraud data:")
            print(self.fraud_df[['transaction_amount']].head())

        else:
            print("Please provide a valid data_type ('creditcard' or 'fraud').")
    def encode_categorical_features(self):
        """Encode categorical features in fraud_df."""
        label_encoder = LabelEncoder()
        self.fraud_df['category'] = label_encoder.fit_transform(self.fraud_df['category'])

        print("Categorical features encoded:")
        print(self.fraud_df[['category']].head())
    def merge_datasets_for_geolocation(self):
        """Convert IP addresses to integer format and merge fraud_df with ip_country_df."""
        # Convert IP addresses to integer format in fraud_df
        self.fraud_df['ip_address_int'] = self.fraud_df['ip_address'].apply(self.convert_ip_to_int)
            # Convert IP address ranges to integer format in ip_country_df
        self.ip_country_df['lower_bound_ip_int'] = self.ip_country_df['lower_bound_ip_address'].apply(self.convert_ip_to_int)
        self.ip_country_df['upper_bound_ip_int'] = self.ip_country_df['upper_bound_ip_address'].apply(self.convert_ip_to_int)
            
            # Merge fraud_df with ip_country_df based on the IP address falling within the country range
        self.fraud_df = pd.merge(
                self.fraud_df,
                self.ip_country_df,
                how='left',
                left_on='ip_address_int',
                right_on='lower_bound_ip_int'
            )
            
            # Keep only rows where the IP address lies within the upper and lower bounds
        self.fraud_df = self.fraud_df[
                (self.fraud_df['ip_address_int'] >= self.fraud_df['lower_bound_ip_int']) & 
                (self.fraud_df['ip_address_int'] <= self.fraud_df['upper_bound_ip_int'])
            ]
            
        print(f"Datasets merged. Fraud dataset now has geolocation information.")