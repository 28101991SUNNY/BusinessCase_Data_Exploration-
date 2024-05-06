from flask import Flask, jsonify, request # type: ignore
import pickle
import pandas as pd # type: ignore

app = Flask(__name__)

with open('model_and_mappings.pkl', 'rb') as file:
    model, mappings = pickle.load(file)

FEATURE_ORDER = [
    "Occupation",
    "Monthly Income",
    "Credit Score",
    "Years of Employment",
    "Finance Status",
    "Finance History",
    "Number of Children"
]

def predict_with_mapping(input_data, model, mappings):
    # Ensure input data follows the predefined feature order
    ordered_input = {feature: input_data[feature] for feature in FEATURE_ORDER}
    
    input_df = pd.DataFrame.from_dict(ordered_input, orient='index').T

    encoded_data = pd.DataFrame(index=input_df.index)

    for column in input_df.columns:
        if column in mappings:
            encoded_data[column] = input_df[column].map(mappings[column])
        else:
            encoded_data[column] = input_df[column]

    prediction = model.predict(encoded_data)

    return prediction[0]

@app.route("/ping", methods=["GET"])
def ping():
    return {"message": "Pinging the NEWWWW model successful!!"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = predict_with_mapping(data, model, mappings)

    if prediction == 0:
        prediction = "Not Approved"
    elif prediction == 1:
        prediction = "Approved"

    return {'prediction': prediction}

