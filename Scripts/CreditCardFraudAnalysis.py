import ipaddress
import pandas as pd
class CreditCardFraudAnalysis:
    def __init__(self, fraud_df: pd.DataFrame, ip_data_df: pd.DataFrame, file_path: str):
        self.fraud_df = fraud_df
        self.ip_data_df = ip_data_df
        self.file_path = file_path  # Store the file path as an instance variable

    def convert_ip_to_numeric(self):
        # Function to safely convert IP addresses
        def safe_convert_ip(ip):
            if pd.isnull(ip):
                return None
            try:
                return int(ipaddress.ip_address(ip))
            except ValueError:
                print(f"Invalid IP address: {ip}")
                return None

        # Apply conversion with validation
        self.fraud_df['ip_numeric'] = self.fraud_df['ip_address'].apply(safe_convert_ip)

        # Convert IP Address Data bounds to numeric
        self.ip_data_df['lower_bound_numeric'] = self.ip_data_df['lower_bound_ip_address'].apply(safe_convert_ip)
        self.ip_data_df['upper_bound_numeric'] = self.ip_data_df['upper_bound_ip_address'].apply(safe_convert_ip)

    def is_valid_ip(self, ip):
        """Check if the given string is a valid IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
            
    def merge_fraud_and_ip_data(self):
        """Merge fraud data with IP address data based on IP numeric values."""
        # Initialize the country column in the fraud DataFrame
        self.fraud_df['country'] = None

        # Create a new DataFrame to hold results
        matched_rows = []

        # Iterate over each fraud record
        for index, fraud_row in self.fraud_df.iterrows():
            fraud_ip_numeric = fraud_row['ip_numeric']
            
            # Skip if the fraud IP numeric is None
            if fraud_ip_numeric is None:
                continue

            # Filter IP data based on current fraud IP numeric
            filtered_ip_data = self.ip_data_df[(
                self.ip_data_df['lower_bound_numeric'].notnull()) &
                (self.ip_data_df['upper_bound_numeric'].notnull()) &
                (self.ip_data_df['lower_bound_numeric'] <= fraud_ip_numeric) &
                (fraud_ip_numeric <= self.ip_data_df['upper_bound_numeric'])
            ]

            # If there are matching IP rows, append to the list
            for _, ip_row in filtered_ip_data.iterrows():
                matched_rows.append({
                    'fraud_index': index,
                    'country': ip_row['country']
                })

        # Update the country column in fraud_df based on matches
        for match in matched_rows:
            self.fraud_df.at[match['fraud_index'], 'country'] = match['country']

        # Log how many matches were found
        print(f"Total fraud cases with matched countries: {self.fraud_df['country'].notnull().sum()}")

    def save_merged_data(self):
        """Save the merged fraud data to a CSV file."""
        self.fraud_df.to_csv(self.file_path, index=False)  # Use the instance variable
        print(f"Merged data saved to {self.file_path}")

    def fill_missing_countries(self):
        """Fill missing country values."""
        self.fraud_df['country'].fillna('Unknown', inplace=True)