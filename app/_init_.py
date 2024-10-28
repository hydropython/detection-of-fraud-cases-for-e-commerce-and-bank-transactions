from flask import Flask

app = Flask(__name__)  # Initialize the Flask application

from app import routes  # Import routes after app is initialized