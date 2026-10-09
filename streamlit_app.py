import streamlit as st

st.set_page_config(
    page_title="ERGO Claim Management",
    page_icon="🏥",
    layout="wide"
)

st.title("ERGO Claim Management")
st.write("AI-powered Claim Analysis & Recommendation")

st.divider()

st.subheader("Claim Information")

claim_id = st.text_input(
    "Claim ID",
    value="CLM1001"
)

policy_number = st.text_input(
    "Policy Number",
    value="PLO1001"
)

customer_name = st.text_input(
    "Customer Name",
    value="Shri Gondkar"
)

claim_type = st.selectbox(
    "Claim Type",
    ["Motor", "Property", "Liability", "Personal Accident"]
)

claim_amount = st.number_input(
    "Claim Amount",
    min_value=0,
    value=85000
)

if st.button("Process Claim", type="primary"):

    st.success("Claim submitted successfully!")

    st.write("### Claim Details")

    st.json({
        "claim_id": claim_id,
        "policy_number": policy_number,
        "customer": customer_name,
        "claim_type": claim_type,
        "claim_amount": claim_amount
    })