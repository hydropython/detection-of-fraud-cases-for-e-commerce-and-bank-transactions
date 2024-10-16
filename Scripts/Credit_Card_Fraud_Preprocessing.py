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

    def handle_missing_values(self, strategy='mean'):
        """Handle missing values in the creditcard_df and fraud_df."""
        imputer = SimpleImputer(strategy=strategy)
        self.creditcard_df = pd.DataFrame(imputer.fit_transform(self.creditcard_df), columns=self.creditcard_df.columns)
        self.fraud_df = pd.DataFrame(imputer.fit_transform(self.fraud_df), columns=self.fraud_df.columns)

        print("Missing values handled using strategy:", strategy)
        print("Creditcard DataFrame - Missing Values:", self.creditcard_df.isnull().sum().sum())
        print("Fraud DataFrame - Missing Values:", self.fraud_df.isnull().sum().sum())

    def data_cleaning(self):
        """Remove duplicates and correct data types."""
        # Remove duplicates
        creditcard_initial_size = len(self.creditcard_df)
        fraud_initial_size = len(self.fraud_df)
        
        self.creditcard_df.drop_duplicates(inplace=True)
        self.fraud_df.drop_duplicates(inplace=True)
        
        print(f"Duplicates removed - Creditcard dataset: {creditcard_initial_size - len(self.creditcard_df)}")
        print(f"Duplicates removed - Fraud dataset: {fraud_initial_size - len(self.fraud_df)}")

        # Correct data types
        self.creditcard_df['Time'] = pd.to_numeric(self.creditcard_df['Time'], errors='coerce')
        self.creditcard_df['Amount'] = pd.to_numeric(self.creditcard_df['Amount'], errors='coerce')
    
    def eda(self):
        """Perform Univariate and Bivariate Analysis with modern visualizations."""
        print("Univariate Analysis:")
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

        print("Bivariate Analysis completed and images saved.")

    def merge_datasets_for_geolocation(self):
        """Merge Fraud_Data with IPAddress_to_Country using IP."""
        self.fraud_df['ip'] = self.fraud_df['ip_address'].apply(self.convert_ip_to_int)
        merged_df = pd.merge(self.fraud_df, self.ip_country_df, how='left', left_on='ip', right_on='ip_from')
        print("Merged Fraud_Data with IPAddress_to_Country. Resulting DataFrame shape:", merged_df.shape)
        return merged_df

    def convert_ip_to_int(self, ip_str):
        """Convert an IP address to an integer."""
        octets = ip_str.split('.')
        return int(octets[0]) * (256 ** 3) + int(octets[1]) * (256 ** 2) + int(octets[2]) * 256 + int(octets[3])

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

    def normalization_scaling(self, method='standard'):
        """Normalize and scale numerical features."""
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.creditcard_df[['Amount', 'Time']] = scaler.fit_transform(self.creditcard_df[['Amount', 'Time']])

        print(f"Data scaled using {method} scaling.")
        print("First few rows of the scaled data:")
        print(self.creditcard_df[['Amount', 'Time']].head())

    def encode_categorical_features(self):
        """Encode categorical features in fraud_df."""
        label_encoder = LabelEncoder()
        self.fraud_df['category'] = label_encoder.fit_transform(self.fraud_df['category'])

        print("Categorical features encoded:")
        print(self.fraud_df[['category']].head())