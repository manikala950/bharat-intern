import pickle
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from the HTML form
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            # Make predictions using the loaded model
            prediction = model.predict([[free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

            return render_template('result.html', prediction=f'Predicted Quality: {prediction[0]:.2f}')
        except Exception as e:
            return render_template('index.html', error=f'Error: {e}')

if __name__ == '__main':
    app.run(debug=True)
