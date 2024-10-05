import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


dataset_path = r'C:\Users\AVINASH\Downloads\Diabetes-Detection-using-Logistic-Regression-main\Diabetes-Detection-using-Logistic-Regression-main\diabetes.csv'
dataset = pd.read_csv(dataset_path)


X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


model_and_scaler = {'model': model, 'scaler': scaler}
joblib.dump(model_and_scaler, 'model_with_scaler.pkl')


try:
    loaded_model_and_scaler = joblib.load('model_with_scaler.pkl')
    loaded_model = loaded_model_and_scaler['model']
    loaded_scaler = loaded_model_and_scaler['scaler']
    print("Model and scaler loaded successfully")
    
    
    sample_data = X_test[0:1] 
    scaled_sample_data = loaded_scaler.transform(sample_data)
    prediction = loaded_model.predict(scaled_sample_data)
    print(f"Sample prediction: {prediction}")
except Exception as e:
    print(f"Error loading model and scaler: {e}")
