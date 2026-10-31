
import os
import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_package_model_v1.joblib"
)

model = joblib.load(model_path)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Tourism Package Purchase Prediction")

st.write("""
This application predicts whether a customer is likely to
**purchase the tourism package** based on customer
characteristics, travel preferences, and sales interaction
details.

Please enter the customer details below to get a prediction.
""")


# ---------------------------------------------------------
# Customer Information
# ---------------------------------------------------------

st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Self Employed",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Single",
        "Divorced"
    ]
)


# ---------------------------------------------------------
# Visit Information
# ---------------------------------------------------------

st.header("Visit Information")

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=2,
    step=1
)

passport = st.selectbox(
    "Has Passport?",
    [
        0,
        1
    ],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

own_car = st.selectbox(
    "Owns a Car?",
    [
        0,
        1
    ],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# ---------------------------------------------------------
# Sales / Pitch Information
# ---------------------------------------------------------

st.header("Sales Interaction")

type_of_contact = st.selectbox(
    "Type of Contact",
    [
        "Self Enquiry",
        "Company Invited"
    ]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    max_value=120,
    value=30,
    step=1
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)


# ---------------------------------------------------------
# Product Information
# ---------------------------------------------------------

st.header("Tourism Product")

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star Rating",
    [3, 4, 5]
)

designation = st.selectbox(
    "Customer Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    max_value=1000000.0,
    value=25000.0,
    step=1000.0
)


# ---------------------------------------------------------
# Assemble input data
# ---------------------------------------------------------

input_data = pd.DataFrame([{

    "Age": age,

    "TypeofContact": type_of_contact,

    "CityTier": city_tier,

    "DurationOfPitch": duration_of_pitch,

    "Occupation": occupation,

    "Gender": gender,

    "NumberOfPersonVisiting": number_of_person_visiting,

    "NumberOfFollowups": number_of_followups,

    "ProductPitched": product_pitched,

    "PreferredPropertyStar": preferred_property_star,

    "MaritalStatus": marital_status,

    "NumberOfTrips": number_of_trips,

    "Passport": passport,

    "PitchSatisfactionScore": pitch_satisfaction_score,

    "OwnCar": own_car,

    "NumberOfChildrenVisiting": number_of_children_visiting,

    "Designation": designation,

    "MonthlyIncome": monthly_income

}])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Tourism Package Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the tourism package."
        )

    else:

        st.warning(
            "Customer is unlikely to purchase the tourism package."
        )

    st.write(
        f"Purchase Probability: **{probability:.2%}**"
    )
