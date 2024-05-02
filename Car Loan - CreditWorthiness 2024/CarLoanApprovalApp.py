import streamlit as st
import pandas as pd
import pickle

def load_model_and_mappings(model_path='model_and_mappings.pkl'):
    with open(model_path, 'rb') as file:
        model, mappings = pickle.load(file)
    return model, mappings

def predict_with_mapping(input_data, model, mappings):
    encoded_data = pd.DataFrame(index=input_data.index)

    for column in input_data.columns:
        if column in mappings:
            encoded_data[column] = input_data[column].map(mappings[column])
        else:
            encoded_data[column] = input_data[column]
   
    predictions = model.predict(encoded_data)
    
    return predictions

def main():
    st.title("Car Worthiness Prediction")

    model, mappings = load_model_and_mappings()

    # Create input fields
    occupation = st.selectbox("Occupation", list(mappings["Occupation"].keys()))
    monthly_income = st.number_input("Monthly Income", value=6000, min_value=0, max_value=150000)
    credit_score = st.number_input("Credit Score", value=800, min_value=300, max_value=800)
    years_of_employment = st.number_input("Years of Employment", value=5)
    finance_status = st.selectbox("Finance Status", list(mappings["Finance Status"].keys()))
    finance_history = st.selectbox("Finance History", list(mappings["Finance History"].keys()))
    num_children = st.number_input("Number of Children", value=0, min_value=0)

    if st.button("Predict"):
        error = False
        error_messages = []

        if monthly_income < 0 or monthly_income > 150000:
            error = True
            error_messages.append("Monthly Income should be between 0 and 150000")
        
        if credit_score < 300 or credit_score > 800:
            error = True
            error_messages.append("Credit Score should be between 300 and 800")
        
        if num_children < 0:
            error = True
            error_messages.append("Number of Children cannot be negative")
        
        if error:
            st.error("\n".join(error_messages))
        else:
            input_data = pd.DataFrame({
                'Occupation': [occupation],
                'Monthly Income': [monthly_income],
                'Credit Score': [credit_score],
                'Years of Employment': [years_of_employment],
                'Finance Status': [finance_status],
                'Finance History': [finance_history],
                'Number of Children': [num_children]
            })

            prediction = predict_with_mapping(input_data, model, mappings)
            
            st.write("Prediction:", prediction)
if __name__ == "__main__":
    main()

