# scripts/equity.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def identify_affordable_quality_stays(df, save_path="underserved_recommendations.pkl"):
    """
    Filter for affordable + high-rated listings in low-income ZIP codes.
    Saves a pickled DataFrame for use in Streamlit.
    """
    city_avg_income = df["INCOME_HOUSEHOLD_MEDIAN"].mean()
    df["underserved_zip"] = df["INCOME_HOUSEHOLD_MEDIAN"] < city_avg_income

    underserved_df = df[df["underserved_zip"]].copy()
    zip_medians = underserved_df.groupby("zipcode")["price_per_person"].median().to_dict()
    underserved_df["zip_median_price"] = underserved_df["zipcode"].map(zip_medians)

    underserved_df["is_affordable"] = underserved_df["price_per_person"] < underserved_df["zip_median_price"]
    underserved_df["is_high_quality"] = underserved_df["review_scores_rating"] >= 4.5

    recommended = underserved_df[
        underserved_df["is_affordable"] & underserved_df["is_high_quality"]
    ][[
        "id", "zipcode", "price", "price_per_person", "review_scores_rating", "amenity_count",
        "room_type", "property_type", "beds", "accommodates", "INCOME_HOUSEHOLD_MEDIAN"
    ]].copy()

    recommended["value_score"] = recommended["review_scores_rating"] / recommended["price_per_person"]
    recommended.sort_values(by="value_score", ascending=False, inplace=True)

    recommended.to_pickle(save_path)
    return recommended

def compute_zipcode_equity_score(df, listings_df, score_csv="zip_equity_scores.csv", recommended_csv="equity_recommended_listings.csv"):
    """
    Compute equity score by ZIP code and flag top 20% for fairness-oriented promotion.
    """
    zip_stats = listings_df.groupby("zipcode").agg({
        "INCOME_HOUSEHOLD_MEDIAN": "mean",
        "price": "mean",
        "review_scores_rating": "mean",
        "id": "count"
    }).reset_index()

    zip_stats.rename(columns={
        "id": "listing_count",
        "INCOME_HOUSEHOLD_MEDIAN": "median_income",
        "price": "avg_price",
        "review_scores_rating": "avg_rating"
    }, inplace=True)

    scaler = MinMaxScaler()
    zip_stats[['norm_income', 'norm_price', 'norm_rating']] = scaler.fit_transform(
        zip_stats[['median_income', 'avg_price', 'avg_rating']]
    )

    # Equity score = favor low income + low price + high quality
    zip_stats['equity_score'] = (
        (1 - zip_stats['norm_income']) * 0.4 +
        (1 - zip_stats['norm_price']) * 0.3 +
        zip_stats['norm_rating'] * 0.3
    )

    zip_stats_sorted = zip_stats.sort_values(by='equity_score', ascending=False)
    top_zips = zip_stats_sorted.head(int(0.2 * len(zip_stats_sorted)))

    equity_listings = listings_df[listings_df['zipcode'].isin(top_zips['zipcode'])]
    zip_stats_sorted.to_csv(score_csv, index=False)
    equity_listings.to_csv(recommended_csv, index=False)

    return zip_stats_sorted, equity_listings
