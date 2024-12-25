import streamlit as st
import pickle
import pandas as pd

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Housing Price Prediction", page_icon="🏠", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            color: #FF6347;
            font-size: 40px;
            text-align: center;
        }
        .header {
            font-size: 30px;
            color: #008080;
            text-align: center;
        }
        .predict-button>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            border-radius: 12px;
            padding: 10px 24px;
        }
        .predict-button>button:hover {
            background-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #F0F4F8;
        }
        /* Background color for the entire page */
        body {
            background-color: #FFF8DC;
        }
        .input-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .input-container div {
            margin: 10px;
            width: 48%;
        }
        /* Centering the prediction results */
        .prediction-box {
            text-align: center;
            background-color: #000000;
            border: 2px solid #FF6347;
            padding: 20px;
            font-weight: bold;
            font-size: 22px;
            width: 60%;
            margin: 30px auto;
        }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
def load_model():
    with open('rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Page content
st.markdown('<p class="main-title">Welcome to Housing Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="header">Please enter the necessary information below to predict the house price.</p>', unsafe_allow_html=True)

# Create a container for inputs
with st.form(key='input_form'):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31)
        housing_median_age = st.number_input("Median age of a house within a block", min_value=1, max_value=52)
        total_rooms = st.number_input("Total number of rooms within a block", min_value=2, max_value=39320)
        population = st.number_input("Total number of people residing within a block", min_value=3, max_value=35682)
        ocean_proximity = st.selectbox("Location of the house w.r.t ocean/sea", ["Near Bay", "Inland", "<1H Ocean", "Near Ocean", "Other"])

    with col2:
        latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95)
        total_bedrooms = st.number_input("Total number of bedrooms within a block", min_value=1, max_value=6445)
        households = st.number_input("Total number of households, a group of people residing within a home unit, for a block", min_value=1, max_value=6082)
        median_income = st.number_input("Median income for households within a block of houses (measured in tens of thousands of US Dollars)", min_value=0.4999, max_value=15.000)

    # Mapping location to numeric value
    location_value = {"Near Bay": 1, "Inland": 2, "<1H Ocean": 3, "Near Ocean": 4, "Other": 5}[ocean_proximity]

    # Submit button
    submit_button = st.form_submit_button(label="Predict House Price")

    # Prediction logic
    if submit_button:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame(
            [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, location_value]],
            columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "new_ocean_proximity"]
        )
        
        # Reorder the input data columns to match the order of features in the model
        input_data = input_data[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_data)
        
        # Display the result in a styled box
        st.markdown(f'<div class="prediction-box">The predicted house price is: <strong>${prediction[0]:,.2f}</strong></div>', unsafe_allow_html=True)

# Run the Streamlit page
if __name__ == "__main__":
    st.write("Wanna buy one?")
