from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('credit.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from input box (comma-separated)
        raw_features = request.form['features']
        input_list = list(map(float, raw_features.split(',')))

        # Ensure it's a 2D array
        features_array = np.array([input_list])

        # Make prediction
        prediction = model.predict(features_array)

        result = "Fraudulent transaction" if prediction[0] == 1 else "Legitimate transaction"

        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
