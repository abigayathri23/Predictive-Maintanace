# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Example model

app = Flask(__name__)

# Initialize your machine learning model (example model)
model = RandomForestRegressor(n_estimators=100, random_state=42)
X_train = np.random.rand(100, 4)  # Example training data
y_train = np.random.rand(100)     # Example training labels
model.fit(X_train, y_train)

# Route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # This serves your HTML file

# Route for processing predictions (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the HTML
    sensor_1 = float(request.form['sensor_1'])
    sensor_2 = float(request.form['sensor_2'])
    sensor_3 = float(request.form['sensor_3'])
    operational_hours = float(request.form['operational_hours'])

    # Create a feature array to feed into the model
    features = np.array([[sensor_1, sensor_2, sensor_3, operational_hours]])

    # Make a prediction using the model
    prediction = model.predict(features)

    # Return the result as JSON (or render a new page)
    return jsonify({'RUL Prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
