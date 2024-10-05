import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and scaler
model = None
scaler = None
model_path = r"C:\Users\AVINASH\Downloads\Diabetes-Detection-using-Logistic-Regression-main\Diabetes-Detection-using-Logistic-Regression-main\model_with_scaler.pkl"

try:
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']
    print("Model and scaler loaded successfully")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def login():
    return render_template("welcome.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about_diabetes')
def about_diabetes():
    return render_template("about_diabetes.html")

@app.route('/working')
def working():
    return render_template("working.html")

@app.route('/OurTeam')
def OurTeam():
    return render_template("OurTeam.html")

@app.route('/userinput', methods=['GET', 'POST'])
def userinput():
    if request.method == 'POST':
        if model is None or scaler is None:
            print("Model or scaler not loaded correctly.")
            return "Model or scaler not loaded correctly", 500

        try:
            # Collect input data from form
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            blood_pressure = int(request.form['blood_pressure'])
            skin_thickness = int(request.form['skin_thickness'])
            insulin = int(request.form['insulin'])
            BMI = float(request.form['BMI'])
            DPF = float(request.form['DPF'])
            age = int(request.form['age'])

            # Prepare the data
            data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, BMI, DPF, age]])

            # Scale the data
            scaled_data = scaler.transform(data)

            # Get prediction and probability
            answer = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)[0][1]  # Probability of being diabetic (class 1)

            # Convert probability to percentage
            diabetes_percentage = probability * 100  # Use the model's predicted probability for percentage

            if answer == 0:
                return render_template("congratulations.html")
            else:
                return render_template("alert.html", diabetes_percentage=diabetes_percentage)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "An error occurred during prediction", 500

    return render_template("userinput.html")

if __name__ == "__main__":
    app.run(debug=True)
