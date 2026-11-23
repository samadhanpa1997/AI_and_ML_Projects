import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")

st.markdown("---")

algorithm = st.selectbox(
    "Select Algorithm",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)

if algorithm == "Logistic Regression":

    model = pickle.load(
        open(
            "models/logistic_regression_model.pkl",
            "rb"
        )
    )

elif algorithm == "Decision Tree":

    model = pickle.load(
        open(
            "models/decision_tree_model.pkl",
            "rb"
        )
    )

else:

    model = pickle.load(
        open(
            "models/random_forest_model.pkl",
            "rb"
        )
    )

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate",
         "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes","No"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=0
    )

    credit_history = st.selectbox(
        "Credit History",
        [0,1]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )

# Encoding

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

education = 1 if education == "Graduate" else 0

self_employed = 1 if self_employed == "Yes" else 0

dependents_map = {
    "0":0,
    "1":1,
    "2":2,
    "3+":3
}

dependents = dependents_map[dependents]

property_map = {
    "Urban":2,
    "Semiurban":1,
    "Rural":0
}

property_area = property_map[property_area]

input_df = pd.DataFrame({

    "Gender":[gender],
    "Married":[married],
    "Dependents":[dependents],
    "Education":[education],
    "Self_Employed":[self_employed],
    "ApplicantIncome":[applicant_income],
    "CoapplicantIncome":[coapplicant_income],
    "LoanAmount":[loan_amount],
    "Loan_Amount_Term":[loan_term],
    "Credit_History":[credit_history],
    "Property_Area":[property_area]

})

if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:

        st.success(
            "✅ Loan Approved"
        )

    else:

        st.error(
            "❌ Loan Rejected"
        )

    st.subheader("Input Data")

    st.dataframe(input_df)