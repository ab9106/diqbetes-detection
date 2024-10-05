import joblib

model_path = r"C:\Users\AVINASH\Downloads\Diabetes-Detection-using-Logistic-Regression-main\Diabetes-Detection-using-Logistic-Regression-main\model.pkl"

try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}.")
except Exception as e:
    print(f"Error loading model: {e}")
