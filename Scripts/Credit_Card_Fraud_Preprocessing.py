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
        # Load datasets
        self.creditcard_df = self.load_data('../Data/creditcard.csv')
        self.fraud_df = self.load_data('../Data/Fraud_Data.csv')
        self.ip_country_df = self.load_data('../Data/IpAddress_to_Country.csv')
        # Ensure the output directory exists
        os.makedirs('../Image', exist_ok=True)
        self.merged_fraud_df = None  # Initialize the merged DataFrame

    def load_data(self, file_path):
        """Load a CSV file into a DataFrame if it exists."""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
            return None

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

    def save_cleaned_data(self, data_type=None, file_name='cleaned_file_name.csv'):
        """Save the cleaned DataFrame to a CSV file."""
        if data_type == 'creditcard':
            self.creditcard_df.to_csv(f'../Data/{file_name}', index=False)
            print(f"Cleaned creditcard_df saved as {file_name} in ../Data/")
        elif data_type == 'fraud':
            self.fraud_df.to_csv(f'../Data/{file_name}', index=False)
            print(f"Cleaned fraud_df saved as {file_name} in ../Data/")
        else:
            print("Please provide a valid data_type ('creditcard' or 'fraud').")