from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly_charges = float(request.form["monthly_charges"])
    contract = int(request.form["contract"])
    senior_citizen = int(request.form["senior_citizen"])
    internet_fiber = int(request.form["internet_fiber"])
    internet_no = int(request.form["internet_no"])
    pay_credit = int(request.form["pay_credit"])
    pay_electronic = int(request.form["pay_electronic"])

    input_df = pd.DataFrame({
        'tenure': [tenure],
        'monthly_charges': [monthly_charges],
        'contract': [contract],
        'senior_citizen': [senior_citizen],
        'internet_service_Fiber': [internet_fiber],
        'internet_service_No': [internet_no],
        'payment_method_Credit card': [pay_credit],
        'payment_method_Electronic check': [pay_electronic]
    })

    # Scaling only numeric continuous
    input_df[['tenure', 'monthly_charges']] = scaler.transform(
        input_df[['tenure', 'monthly_charges']]
    )

    prediction = model.predict(input_df)[0]

    result = "Customer will Churn ❌" if prediction == 1 else "Customer will Stay ✅"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)