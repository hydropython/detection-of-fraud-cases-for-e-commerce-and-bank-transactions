from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio

# Create a Flask app instance
app = Flask(__name__)
# Initialize the global variable
balanced_fraud_df = None
# Load the trained model
model_path = os.path.join('models', 'gradient_boosting_model_fraud.pkl')
gradient_boosting_model = joblib.load(model_path)

# Global variable to store the uploaded dataset
balanced_fraud_df = None

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for uploading the dataset
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    global balanced_fraud_df  # Declare the global variable
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file and uploaded_file.filename != '':
            # Save the uploaded file to a specified location
            file_path = os.path.join('data', uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Load the uploaded dataset
            balanced_fraud_df = pd.read_csv(file_path)
            print(balanced_fraud_df.head())  # Debugging line to check the DataFrame
            
            # Redirect to the dashboard after uploading
            return redirect(url_for('dashboard'))

        else:
            return "<h3>No file selected or the file is empty.</h3>"

    return render_template('dataset.html')

# EDA Plotting Functions
def create_transaction_distribution_plot(df):
    fig = px.histogram(df, x='purchase_value', color='class',
                       title='Purchase Value Distribution',
                       labels={'class': 'Fraudulent Transaction'})
    return pio.to_html(fig, full_html=False)

def create_fraud_rate_plot(df):
    fraud_rate = df.groupby('signup_time')['class'].mean().reset_index()
    fig = px.line(fraud_rate, x='signup_time', y='class',
                  title='Fraud Rate Over Time',
                  labels={'class': 'Fraud Rate'})
    return pio.to_html(fig, full_html=False)

def create_fraud_by_country_plot(df):
    fraud_count = df.groupby('country')['class'].sum().reset_index()
    fig = px.bar(fraud_count, x='country', y='class',
                 title='Fraud Count by Country',
                 labels={'class': 'Fraud Count'})
    return pio.to_html(fig, full_html=False)

def create_correlation_matrix_plot(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    fig = px.imshow(correlation,
                    title='Correlation Matrix',
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=correlation.columns,
                    y=correlation.columns)
    return pio.to_html(fig, full_html=False)

def create_fraud_by_category_plot(df):
    if 'category' in df.columns:
        fraud_count = df.groupby('category')['class'].sum().reset_index()
        title = 'Fraud Count by Category'
    elif 'transaction_type' in df.columns:
        fraud_count = df.groupby('transaction_type')['class'].sum().reset_index()
        title = 'Fraud Count by Transaction Type'
    else:
        return "<div>No category or transaction type data available for plotting.</div>"
    
    fig = px.bar(fraud_count, x=fraud_count.columns[0], y='class',
                 title=title, labels={'class': 'Fraud Count'})
    return pio.to_html(fig, full_html=False)

def create_transaction_type_distribution_plot(df):
    if 'transaction_type' in df.columns:
        fig = px.pie(df, names='transaction_type', values='class',
                     title='Transaction Type Distribution')
        return pio.to_html(fig, full_html=False)
    else:
        return "<div>No transaction type data available for plotting.</div>"

def create_geographical_fraud_plot(df):
    if 'country' in df.columns and 'class' in df.columns:
        # Count fraudulent transactions per country
        fraud_count_by_country = df.groupby('country')['class'].sum().reset_index()
        fraud_count_by_country.columns = ['Country', 'Fraud Count']  # Renaming for clarity

        # Create a geographical plot using Plotly Express
        fig = px.choropleth(
            fraud_count_by_country,
            locations='Country',
            locationmode='country names',  # Ensure the country names are recognized
            color='Fraud Count',
            title='Geographical Distribution of Fraudulent Transactions',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={'Fraud Count': 'Fraud Count'},
            template='plotly_dark'  # Optional: Use dark template for aesthetics
        )

        return pio.to_html(fig, full_html=False)
    else:
        return "<div>No geographical data available for plotting.</div>"

def create_monthly_fraud_trends_plot(df):
    # Ensure that the 'signup_time' column is in datetime format
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    
    # Group by month and sum the fraud cases
    monthly_fraud = df.groupby(df['signup_time'].dt.to_period('M'))['class'].sum().reset_index()
    
    # Convert the period back to datetime for plotting
    monthly_fraud['signup_time'] = monthly_fraud['signup_time'].dt.to_timestamp()
    
    # Create the plot
    fig = px.line(monthly_fraud, x='signup_time', y='class', 
                  title='Monthly Fraud Trends',
                  labels={'signup_time': 'Month', 'class': 'Number of Fraud Cases'})
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

@app.route('/dashboard')
def dashboard():
    global balanced_fraud_df  # Use the global variable for the DataFrame
    if balanced_fraud_df is not None:
        # Generate plots using the dataset
        plots = {
            'plot1': create_transaction_distribution_plot(balanced_fraud_df),
            'plot2': create_fraud_rate_plot(balanced_fraud_df),
            'plot3': create_correlation_matrix_plot(balanced_fraud_df),
            'plot4': create_fraud_by_category_plot(balanced_fraud_df),
            'plot5': create_transaction_type_distribution_plot(balanced_fraud_df),
            'plot6': create_geographical_fraud_plot(balanced_fraud_df),
            'plot7': create_fraud_by_country_plot(balanced_fraud_df),
            'plot8': create_monthly_fraud_trends_plot(balanced_fraud_df)
        }
        # Include the shape of the DataFrame for additional context
        return render_template('dashboard.html', balanced_fraud_df=balanced_fraud_df, **plots)
    else:
        return "<h3>Please upload a dataset first.</h3>"



# Route for creating a transaction
@app.route('/create_transaction', methods=['GET', 'POST'])
def create_transaction():
    if request.method == 'POST':
        # Extract data from the form
        user_id = request.form['user_id']
        signup_time = request.form['signup_time']
        purchase_time = request.form['purchase_time']
        purchase_value = float(request.form['purchase_value'])
        device_id = request.form['device_id']
        source = request.form['source']
        browser = request.form['browser']
        sex = request.form['sex']
        age = int(request.form['age'])
        ip_address = request.form['ip_address']
        country = request.form['country']  # Capture country from the form

        # Here you could convert IP address to numeric if necessary
        ip_numeric = convert_ip_to_numeric(ip_address)  # Implement this function as needed

        # Prepare data for prediction (if using a model)
        transaction_data = {
            'user_id': user_id,
            'signup_time': signup_time,
            'purchase_time': purchase_time,
            'purchase_value': purchase_value,
            'device_id': device_id,
            'source': source,
            'browser': browser,
            'sex': sex,
            'age': age,
            'ip_address': ip_address,
            'ip_numeric': ip_numeric,
            'country': country
        }

        # Call your prediction function here
        prediction = make_prediction(transaction_data)  # Implement this function to handle predictions

        # Render a result page or redirect back with feedback
        return render_template('prediction_result.html', prediction=prediction)

    return render_template('create_transaction.html')  # Render the transaction creation page

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame(data, index=[0])
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=gradient_boosting_model.feature_names_in_, fill_value=0)
    prediction = gradient_boosting_model.predict(input_data)
    result = {'prediction': 'Fraud' if prediction[0] == 1 else 'Not Fraud'}
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)