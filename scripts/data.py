# scripts/data.py

import pandas as pd
import numpy as np

def load_raw_data(listings_path="data/listings.csv", income_path="data/IncomeHouseholdMedian.xlsx"):
    listings = pd.read_csv(listings_path)
    income = pd.read_excel(income_path)
    return listings, income

def clean_listings(listings):
    
    # Fill numerical missing values
    numeric_fill = ['bathrooms', 'bedrooms', 'beds', 'reviews_per_month']
    for col in numeric_fill:
        if col in listings.columns:
            listings[col] = listings[col].fillna(listings[col].median())

    # Fill categorical
    categorical_fill = ['host_response_time', 'neighbourhood_cleansed']
    for col in categorical_fill:
        if col in listings.columns:
            listings[col] = listings[col].fillna("Unknown")

    # Review score filling
    review_cols = [
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]
    listings[review_cols] = listings[review_cols].fillna(0)
    listings['reviews_per_month'] = listings['reviews_per_month'].fillna(0)
    listings['number_of_reviews'] = listings['number_of_reviews'].fillna(0)
    listings['number_of_reviews_ltm'] = listings['number_of_reviews_ltm'].fillna(0)
    listings['number_of_reviews_l30d'] = listings['number_of_reviews_l30d'].fillna(0)

    # Drop rows without price
    listings = listings[listings['price'].notnull()]
    listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

    # Count amenities
    listings['amenity_count'] = listings['amenities'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

    # Drop duplicates
    listings = listings.drop_duplicates()

    return listings

def merge_income(listings, income):
    income['zipcode'] = income['zipcode'].astype(str)
    listings['zipcode'] = listings['zipcode'].astype(str)

    listings = listings.merge(income, on='zipcode', how='left')
    listings['INCOME_HOUSEHOLD_MEDIAN'] = listings['INCOME_HOUSEHOLD_MEDIAN'].fillna(
        listings['INCOME_HOUSEHOLD_MEDIAN'].median()
    )
    return listings

def save_clean_data(df, path="data/listings.csv"):
    df.to_csv(path, index=False)

