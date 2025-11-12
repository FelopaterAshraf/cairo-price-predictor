import os
import joblib
import streamlit as st
import pandas as pd

from src.utils import clean_and_engineer  # ✅ correct import

st.set_page_config(page_title="Cairo Real Estate Price Predictor", layout="centered")

st.title("Apartment Price Predictor")
st.write("Select a trained model and enter apartment details to estimate the price.")

# --- Load available models ---
MODELS_DIR = "models"
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]

if not model_files:
    st.error("No models found. Run train_model.py first.")
    st.stop()

selected_model_file = st.selectbox("Choose a model (name_R2%)", sorted(model_files))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model_path = os.path.join(MODELS_DIR, selected_model_file)
model = load_model(model_path)

st.subheader("Enter Apartment Features")

col1, col2 = st.columns(2)

with col1:
    area_sqm = st.number_input("Area (sqm)", min_value=40, max_value=400, value=150)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
    floor_number = st.number_input("Floor Number", min_value=1, max_value=30, value=3)
    building_age_years = st.number_input("Building Age (years)", min_value=0, max_value=50, value=5)

with col2:
    district = st.selectbox(
    "District",
    [
        "Madinaty",
        "Fifth Settlement",
        "Rehab City",
        "Katameya",
        "New Cairo (Other)"
    ]
)
    compound_name = st.selectbox(
        "Compound / Area",
        [
            "Rehab 1", "Rehab 2", "Rehab 3", "Rehab 4",
            "Madinaty B1", "Madinaty B2", "Madinaty B3", "Madinaty B4",
            "Gardenia Springs", "Hyde Park", "Katameya Dunes", "Katameya Heights",
            "Katameya Plaza", "Lake View", "Mirage City", "Moon Valley",
            "Mountain View", "Palm Hills"
        ]
    )
    distance_to_auc_km = st.number_input("Distance to AUC (km)", min_value=0.0, value=5.0)
    distance_to_mall_km = st.number_input("Distance to Mall (km)", min_value=0.0, value=3.0)
    distance_to_metro_km = st.number_input("Distance to Metro (km)", min_value=0.0, value=8.0)

finishing_type = st.selectbox("Finishing Type", ["Super Lux", "Lux", "Semi-finished", "Unfinished"])

has_balcony = st.checkbox("Balcony", value=True)
has_parking = st.checkbox("Parking", value=True)
has_security = st.checkbox("Security", value=True)
has_amenities = st.checkbox("Amenities", value=True)
is_negotiable = st.checkbox("Negotiable", value=True)

# 🔧 Extra optional inputs for features existing in training
view_type = st.selectbox("View Type", ["Street", "Garden", "Nile", "Unfinished"])
seller_type = st.selectbox("Seller type", ["Owner", "Broker"])
days_on_market = st.number_input("Days on Market", min_value=1, max_value=300, value=60)

# --- Prediction ---
if st.button("Predict Price"):
    input_dict = {
        "area_sqm": area_sqm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor_number": floor_number,
        "building_age_years": building_age_years,
        "district": district,
        "compound_name": compound_name,
        "distance_to_auc_km": distance_to_auc_km,
        "distance_to_mall_km": distance_to_mall_km,
        "distance_to_metro_km": distance_to_metro_km,
        "finishing_type": finishing_type,
        "has_balcony": "yes" if has_balcony else "no",
        "has_parking": "yes" if has_parking else "no",
        "has_security": "yes" if has_security else "no",
        "has_amenities": "yes" if has_amenities else "no",
        "is_negotiable": "yes" if is_negotiable else "no",
        "view_type": view_type,
        "seller_type": seller_type,
        "days_on_market": days_on_market
    }

    df_input = pd.DataFrame([input_dict])

    #  Apply same cleaning/feature engineering as training
    df_input = clean_and_engineer(df_input)

    #  Ensure all expected columns exist (fill missing with 0)
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    predicted_price = model.predict(df_input)[0]

    st.success(f"Estimated Price: **{predicted_price:,.0f} EGP**")
    st.caption(f"Model used: `{selected_model_file}`")


