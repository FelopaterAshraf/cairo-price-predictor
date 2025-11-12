import pandas as pd

def load_data(path="data/cairo_real_estate_dataset.csv"):
    return pd.read_csv(path)

def clean_and_engineer(df):
    df = df.copy()

    # Drop unused columns
    df = df.drop(columns=["listing_id", "listing_date"], errors="ignore")

    # Fill missing compound names
    df["compound_name"] = df["compound_name"].fillna("Unknown")

    # Encode Yes/No columns
    bool_cols = ["has_balcony", "has_parking", "has_security", "has_amenities", "is_negotiable"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)

    # Filter only 2–3 bedroom apartments
    if "bedrooms" in df.columns:
        df = df[df["bedrooms"].isin([2, 3])]

    # Encode finishing type
    finishing_map = {"Super Lux": 3, "Lux": 2, "Semi-finished": 1, "Unfinished": 0}
    if "finishing_type" in df.columns:
        df["finishing_type_encoded"] = df["finishing_type"].map(finishing_map)
        df = df.drop(columns=["finishing_type"])

    # Price per sqm
    if "price_egp" in df.columns and "area_sqm" in df.columns:
        df["price_per_sqm"] = df["price_egp"] / df["area_sqm"]

    # Proximity score
    if all(c in df.columns for c in ["distance_to_auc_km", "distance_to_mall_km", "distance_to_metro_km"]):
        df["proximity_score"] = (
            df["distance_to_auc_km"] + df["distance_to_mall_km"] + df["distance_to_metro_km"]
        )

    # District average price
    if "district" in df.columns and "price_egp" in df.columns:
        district_avg_price = df.groupby("district")["price_egp"].mean().to_dict()
        df["district_avg_price"] = df["district"].map(district_avg_price)

    # Compound quality score
    if "compound_name" in df.columns and "price_egp" in df.columns:
        compound_avg_price = df.groupby("compound_name")["price_egp"].mean().to_dict()
        df["compound_quality_score"] = df["compound_name"].map(compound_avg_price)

    return df

def split_features_target(df):
    X = df.drop(columns=["price_egp"])
    y = df["price_egp"]
    return X, y
