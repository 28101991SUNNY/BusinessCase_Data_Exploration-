import pickle
import pandas as pd

def predict_with_mapping(input_data, model, mappings):
    input_df = pd.DataFrame.from_dict(input_data, orient='index').T

    encoded_data = pd.DataFrame(index=input_df.index)

    for column in input_df.columns:
        if column in mappings:
            encoded_data[column] = input_df[column].map(mappings[column])
        else:
            encoded_data[column] = input_df[column]

    prediction = model.predict(encoded_data)

    return prediction

with open('model_and_mappings.pkl', 'rb') as file:
    model, mappings = pickle.load(file)

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = predict_with_mapping(data, model, mappings)

    if prediction == 0:
        prediction = "Not Approved"
    elif prediction == 1:
        prediction = "Approved"

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
