# scripts/features.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_engineered_features(df):
    """Add new features like price per person, log_price, etc."""
    df['price_per_person'] = df['price'] / df['accommodates']
    df['log_price'] = np.log1p(df['price'])
    df['host_listing_ratio'] = df['calculated_host_listings_count'] / df['calculated_host_listings_count'].max()
    df['is_entire_home'] = df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
    df['host_verification_count'] = df['host_verifications'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    return df

def encode_and_scale(df):
    """One-hot encode categorical and normalize numeric features."""
    df = pd.get_dummies(df, columns=['room_type', 'property_type'], drop_first=True)

    numeric_cols = [
        'price', 'price_per_person', 'log_price', 'INCOME_HOUSEHOLD_MEDIAN',
        'amenity_count', 'number_of_reviews',
        'review_scores_rating', 'review_scores_cleanliness', 'review_scores_value'
    ]

    # Fill missing numeric
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
