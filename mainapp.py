from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load encoders & model
label_dt = pickle.load(open("label_dt.pkl", "rb"))
model = pickle.load(open("car_price_model.pkl", "rb"))


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "firstpage.html",
        car_names=label_dt["Car_Name"],
        fuel_types=label_dt["Fuel_Type"],
        seller_types=label_dt["Seller_Type"],
        transmissions=label_dt["Transmission"]
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Read inputs from form
    car_name = request.form["car_name"]
    year = int(request.form["year"])
    km_driven = int(request.form["km_driven"])
    fuel_type = request.form["fuel_type"]
    seller_type = request.form["seller_type"]
    transmission_type = request.form["transmission"]
    owner = int(request.form["owner"])

    # Convert to encoded indexes
    car_name_idx = label_dt["Car_Name"].tolist().index(car_name)
    fuel_type_idx = label_dt["Fuel_Type"].tolist().index(fuel_type)
    seller_type_idx = label_dt["Seller_Type"].tolist().index(seller_type)
    transmission_idx = label_dt["Transmission"].tolist().index(transmission_type)

    # Create prediction DF
    input_df = pd.DataFrame([[car_name_idx, year, km_driven,
                              fuel_type_idx, seller_type_idx,
                              transmission_idx, owner]])

    # Predict
    result = model.predict(input_df)[0]

    return render_template("result.html", price=result)


import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
