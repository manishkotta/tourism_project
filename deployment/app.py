import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Manish9119/tourism-package-model", 
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("🌴 Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the **Wellness Tourism Package** 
based on their profile and interaction data. Enter the customer details below to get a prediction.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    type_of_contact_encoded = 0 if type_of_contact == "Company Invited" else 1
    
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    occupation_mapping = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
    occupation_encoded = occupation_mapping[occupation]
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_encoded = 0 if gender == "Female" else 1
    
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    marital_status_mapping = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
    marital_status_encoded = marital_status_mapping[marital_status]
    
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    designation_mapping = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
    designation_encoded = designation_mapping[designation]
    
    monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=100000, value=25000, step=1000)

with col2:
    st.subheader("Travel Preferences & Interaction")
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
    
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
    
    preferred_property_star = st.slider("Preferred Property Star Rating", min_value=3.0, max_value=5.0, value=3.0, step=0.5)
    
    num_trips = st.number_input("Number of Trips (per year)", min_value=1, max_value=20, value=2, step=1)
    
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    passport_encoded = 1 if passport == "Yes" else 0
    
    own_car = st.selectbox("Owns Car?", ["No", "Yes"])
    own_car_encoded = 1 if own_car == "Yes" else 0
    
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    product_mapping = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
    product_pitched_encoded = product_mapping[product_pitched]
    
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=60, value=15, step=1)
    
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3, step=1)
    
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

# Assemble input into DataFrame with correct feature order
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact_encoded,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation_encoded,
    'Gender': gender_encoded,
    'NumberOfPersonVisiting': num_persons,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched_encoded,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status_encoded,
    'NumberOfTrips': num_trips,
    'Passport': passport_encoded,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car_encoded,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation_encoded,
    'MonthlyIncome': monthly_income
}])

# Add some spacing
st.write("")

# Predict button
if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.write("---")
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.success(f"✅ **High Likelihood**: Customer is likely to purchase the package!")
            st.metric("Purchase Probability", f"{prediction_proba[1]:.1%}")
        else:
            st.warning(f"⚠️ **Low Likelihood**: Customer may not purchase the package.")
            st.metric("Purchase Probability", f"{prediction_proba[1]:.1%}")
        
        # Additional insights
        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Will Purchase", "Yes" if prediction == 1 else "No")
        with col_b:
            st.metric("Confidence", f"{max(prediction_proba):.1%}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer
st.write("---")
st.caption("🏖️ Powered by Visit with Us | Tourism Package Prediction System")
