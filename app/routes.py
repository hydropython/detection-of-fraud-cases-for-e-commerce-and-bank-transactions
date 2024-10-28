from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
import pandas as pd
import pickle
import logging
import numpy as np
from app import app

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the Gradient Boosting model
model_path = '../models/gradient_boosting_model.pkl'  # Update with the actual path
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        logging.info("Gradient Boosting model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None  # Set model to None if loading fails

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        try:
            transaction_amount = float(request.form['transaction_amount'])
            transaction_time = float(request.form['transaction_time'])
            # Prepare the input data for the model
            input_data = np.array([[transaction_amount, transaction_time]])

            # Make prediction using the loaded model
            if model is not None:
                prediction = model.predict(input_data)
                
                # Flash a message based on the prediction
                if prediction[0] == 1:  # Assuming 1 indicates fraud
                    flash('Fraudulent transaction detected!', 'danger')
                else:
                    flash('Transaction is safe.', 'success')
            else:
                flash('Model is not available for prediction.', 'danger')

            return redirect(url_for('home'))  # Redirect back to home

        except Exception as e:
            flash('Error processing the input: ' + str(e), 'danger')
            return redirect(url_for('home'))

    return render_template('analyze.html')  # If GET, render an analysis form