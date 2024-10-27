import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class FraudDataProcessing:
    def __init__(self, file_path):
        """Initialize the class with the CSV file path."""
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

        # Convert categorical columns to category dtype for efficiency
        categorical_cols = ['device_id', 'source', 'browser', 'sex', 'class']
        for col in categorical_cols:
            self.df[col] = self.df[col].astype('category')

    def feature_engineering(self):
        """Perform feature engineering on the dataset."""
        # Create transaction frequency and velocity features
        freq = self.df.groupby('user_id')['purchase_time'].transform('count')
        self.df['transaction_frequency'] = freq
        self.df['transaction_velocity'] = self.df['purchase_value'] / self.df['transaction_frequency']

        # Create time-based features
        purchase_time = pd.to_datetime(self.df['purchase_time'])
        self.df['hour_of_day'] = purchase_time.dt.hour
        self.df['day_of_week'] = purchase_time.dt.dayofweek
        
    def normalization_and_scaling(self):
        """Normalize and scale numerical features."""
        scaler = StandardScaler()
        
        # List of numerical columns to scale
        numerical_cols = ['purchase_value', 'transaction_frequency', 'transaction_velocity', 'age']
        
        # Apply scaling
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        
        print("Normalization and scaling completed. Here's a preview of the dataset:")
        print(self.df[numerical_cols].head())
        
    def encode_categorical_features(self):
        """Encode categorical features using One-Hot Encoding and frequency encoding."""
        
        # Identify categorical columns for one-hot encoding
        categorical_cols = ['source', 'browser', 'sex', 'class']  # Low cardinality columns
        
        # One-Hot Encoding for smaller cardinality features
        ohe = OneHotEncoder(drop='first', sparse_output=True)
        encoded_features = ohe.fit_transform(self.df[categorical_cols])
        
        # Create a sparse DataFrame for encoded features
        encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_features, columns=ohe.get_feature_names_out(categorical_cols))
        
        # Frequency encoding for 'device_id' and 'country'
        frequency_device_id = self.df['device_id'].value_counts()
        frequency_country = self.df['country'].value_counts()
        
        # Map the frequencies back to the original DataFrame
        self.df['device_id_freq'] = self.df['device_id'].map(frequency_device_id)
        self.df['country_freq'] = self.df['country'].map(frequency_country)
        
        # Concatenate the encoded features with the original DataFrame
        self.df = pd.concat([self.df.drop(columns=categorical_cols + ['device_id', 'country']), encoded_df], axis=1)

        print("Categorical encoding completed. Here's a preview of the dataset:")
        print(self.df.head())

    def save_processed_data(self, output_file_path):
        """Save the processed DataFrame to a CSV file."""
        self.df.to_csv(output_file_path, index=False)

