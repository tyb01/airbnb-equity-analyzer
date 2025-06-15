import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import pickle
from scripts.data import load_raw_data, clean_listings, merge_income
from scripts.features import add_engineered_features, encode_and_scale
from scripts.model import train_model, predict_and_save_undervalued
from scripts.recommend import explain_with_shap, recommend_host_upgrades_verbose
from scripts.equity import identify_affordable_quality_stays, compute_zipcode_equity_score
import seaborn as sns

df = pd.read_csv("data/listings.csv")
features = [
        'accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenity_count',
        'reviews_per_month', 'minimum_nights', 'maximum_nights',
        'availability_365', 'number_of_reviews', 'zipcode', 'INCOME_HOUSEHOLD_MEDIAN',
        'price_per_person', 'host_listing_ratio', 'is_entire_home', 'host_verification_count'
    ]
# ======= CACHED LOADING =======
@st.cache_data

def load_data():
    listings, income = load_raw_data()
    #listings = clean_listings(listings)     DONE ALREADY
    #listings = merge_income(listings, income)      DONE ALREADY
    #listings = add_engineered_features(listings)   DONE ALREADY
    listings_encoded = encode_and_scale(listings.copy())
    return listings, listings_encoded

@st.cache_resource

def load_model():
    
    if not os.path.exists("model.pkl"):
        if df is not None and features is not None:
            print("⚠️ model.pkl not found. Training a new model...")
            model, _ = train_model(df, features)
    return joblib.load("model.pkl")

# ======= MAIN =======
st.set_page_config(layout="wide", page_title="Airbnb Equity Analyzer")
st.title("🏡 Airbnb Equity & Pricing Intelligence Dashboard")

st.sidebar.title("🔎 Navigation")
section = st.sidebar.radio("Go to:", [
    "📖 Introduction",
    "📊 EDA",
    "📈 Model",
    "🧳 Traveler Mode",
    "🧑‍💼 Host Mode",
    "🌍 Communities Mode",
    "📌 Conclusion"
])

# ======= LOAD =======
listings, listings_encoded = load_data()
model = load_model()

# ======= INTRODUCTION =======
if section == "📖 Introduction":
    st.markdown("""
        <style>
            .intro-container {
                background: #2e86c1;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 2rem;
            }
            .intro-title {
                font-size: 2.2rem;
                font-weight: bold;
                color: #003566;
                margin-bottom: 0.8rem;
            }
            .intro-subtext {
                font-size: 1.05rem;
                color: #333;
                margin-bottom: 1.2rem;
            }
        </style>
        <div class="intro-container">
            <div class="intro-title">🎯 Welcome to the Airbnb Equity & Pricing Dashboard</div>
            <div class="intro-subtext">
                This interactive data product helps <strong>Hosts</strong>, <strong>Travelers</strong>, and <strong>Communities</strong> make smarter decisions through intelligent pricing and equity analytics.
                Built with Python, Streamlit, and machine learning models.
            </div>
            <ul>
                <li>🏡 <strong>Hosts</strong> — Optimize your prices and discover amenity upgrades</li>
                <li>🧳 <strong>Travelers</strong> — Discover affordable, high-quality stays</li>
                <li>🌍 <strong>Communities</strong> — Promote fair and equitable tourism across ZIP codes</li>
            </ul>
        </div>
        <hr style="margin: 2rem 0;">

        ### 📂 Dataset Sources
        - `listings.csv`: Airbnb NYC listing data (from https://insideairbnb.com/get-the-data/)
        - `IncomeHouseholdMedian.xlsx`: ZIP-level median household income (from https://simplemaps.com/)

        ### 🚀 How It Works
        - Machine learning (Random Forest) predicts listing prices
        - SHAP analysis interprets feature importance
        - Interactive filters guide both hosts and travelers
        - Equity metrics highlight underserved ZIP codes for better distribution

        ---
        👉 Use the **sidebar** on left to navigate through EDA, host pricing tools, traveler filters, and community-level analysis.
    """, unsafe_allow_html=True)

    st.toast("Welcome! Make data fair and actionable ✨")
    st.balloons()

# ======= EDA =======
elif section == "📊 EDA":
    st.header("🔍 Exploratory Data Analysis")

    st.markdown("Explore the key distributions, correlations, and patterns in the Airbnb dataset using visual insights.")

    # Sample Data
    st.subheader("🧾 Sample of the Dataset")
    st.dataframe(listings.head(10))

    # 💰 Price Distribution
    st.subheader("💰 Price Distribution")
    fig, ax = plt.subplots()
    listings['price'].hist(bins=50, ax=ax, color="#4a90e2")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Number of Listings")
    ax.set_title("Distribution of Listing Prices")
    st.pyplot(fig)
    st.markdown("Most listings are priced below $500, with a long tail of high-end listings.")

    # 🏠 Room Type vs Price
    st.subheader("🏠 Room Type vs Price")
    fig2, ax2 = plt.subplots()
    listings.boxplot(column='price', by='room_type', ax=ax2)
    ax2.set_title("Price Distribution by Room Type")
    ax2.set_ylabel("Price ($)")
    st.pyplot(fig2)
    st.markdown("Entire homes tend to command higher prices compared to private/shared rooms.")


    # 🧭 Median Price by ZIP Code
    st.subheader("📍 Median Price by ZIP Code")
    zip_price = listings.groupby('zipcode')['price'].median().sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    zip_price.head(20).plot(kind='bar', color='#ff7f0e', ax=ax4)
    ax4.set_ylabel("Median Price ($)")
    ax4.set_title("Top 20 ZIP Codes by Median Listing Price")
    st.pyplot(fig4)
    st.markdown("Some ZIP codes consistently have higher pricing due to location or amenities.")

    #Listings Count by Zip Code
    st.subheader("📍 Listings Count by Zip Code")
    zip_counts = listings['zipcode'].value_counts().sort_values(ascending=False)
    st.bar_chart(zip_counts.head(20))
    st.markdown("Higher density in certain zip codes indicates potential competition or popularity.")

    # 📊 Price Distribution by Income Level
    st.subheader("📊 Price vs Household Income Brackets")
    listings['income_bracket'] = pd.qcut(listings['INCOME_HOUSEHOLD_MEDIAN'], q=3, labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots()
    sns.boxplot(x='income_bracket', y='price', data=listings, ax=ax)
    st.pyplot(fig)
    st.markdown("Listings in wealthier neighborhoods tend to be more expensive.")

    # 💵 Income vs Price
    st.subheader("💵 Income vs Price")
    fig6, ax6 = plt.subplots()
    ax6.scatter(listings['INCOME_HOUSEHOLD_MEDIAN'], listings['price'], alpha=0.5, color='purple')
    ax6.set_xlabel("Median Household Income ($)")
    ax6.set_ylabel("Price ($)")
    ax6.set_title("Income vs Price")
    st.pyplot(fig6)
    st.markdown("There is a moderate correlation between income levels and listing price.")

    # 📦 Amenities vs Price
    st.subheader("📦 Amenities vs Price")
    fig7, ax7 = plt.subplots()
    ax7.scatter(listings['amenity_count'], listings['price'], alpha=0.5, color='orange')
    ax7.set_xlabel("Amenity Count")
    ax7.set_ylabel("Price ($)")
    ax7.set_title("Number of Amenities vs Listing Price")
    st.pyplot(fig7)
    st.markdown("Listings with more amenities generally charge higher prices.")

    # 🏢 Property Type Distribution
    st.subheader("🏢 Listing Count by Property Type")
    prop_counts = listings['property_type'].value_counts().head(15)
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    prop_counts.plot(kind='barh', color='#1f77b4', ax=ax8)
    ax8.set_xlabel("Number of Listings")
    ax8.set_title("Top 15 Property Types")
    st.pyplot(fig8)
    st.markdown("Entire apartments and houses are the most common listing types.")

    # 📊 Host Listing Ratio Distribution
    st.subheader("📊 Host Listing Ratio")
    fig9, ax9 = plt.subplots()
    listings['host_listing_ratio'].hist(bins=30, color='#2ca02c', ax=ax9)
    ax9.set_title("Distribution of Host Listing Ratio")
    ax9.set_xlabel("Ratio")
    ax9.set_ylabel("Number of Hosts")
    st.pyplot(fig9)
    st.markdown("Many hosts have a low listing ratio, but a few manage many properties.")

    # 🔗 Correlation Heatmap
    st.subheader("🧠 Correlation Heatmap (Numerical Features)")
    numeric_cols = listings.select_dtypes(include='number').dropna(axis=1).columns
    corr = listings[numeric_cols].corr()
    fig10, ax10 = plt.subplots(figsize=(12, 8))
    import seaborn as sns
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax10)
    ax10.set_title("Correlation Matrix of Numeric Features")
    st.pyplot(fig10)
    st.markdown("Explore which variables move together — useful for feature engineering.")

# ======= MODEL =======
elif section == "📈 Model":
    st.header("📈 Random Forest Pricing Model")

    st.markdown("""
    This section uses a **supervised machine learning model** – specifically a **Random Forest Regressor** – to predict Airbnb listing prices based on key listing features.

    Random Forest is a powerful ensemble algorithm that:
    - Performs well with medium-sized datasets like Airbnb

    We evaluate model performance using **MAE**, **RMSE**, and **R²** to ensure it’s suitable for price prediction.
    """)

    if st.button("🚀 Train Model and Show Evaluation"):
        import time
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        import matplotlib.pyplot as plt

        start_time = time.time()
        model, X_train = train_model(listings, features)
        end_time = time.time()
        st.success(f"✅ Model trained successfully in {end_time - start_time:.2f} seconds.")

        # Evaluation
        y_true = listings['price']
        y_pred = model.predict(listings[features])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        st.subheader("📊 Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${mae:.2f}", "Lower is better")
        col2.metric("RMSE", f"${rmse:.2f}", "Lower is better")
        col3.metric("R² Score", f"{r2:.2%}", "Higher is better")

        st.markdown(f"""
        ✅ The model achieved an **MAE of ${mae:.2f}** and **R² of {r2:.2%}**, meaning it explains about **{r2:.0%} of the price variability**.
        This makes it highly useful for:
        - 🎯 **Hosts** aiming to set data-driven prices
        - 🏙️ **Urban planners** monitoring price equity across neighborhoods
        """)

        # Undervalued Listings
        st.subheader("💡 Top Undervalued Listings")
        undervalued = predict_and_save_undervalued(listings, model, features)
        undervalued['id'] = undervalued['id'].astype(str)
        st.dataframe(undervalued[['id', 'zipcode', 'price', 'predicted_price', 'undervaluation']].head(10))

        # Optional: Actual vs Predicted Plot
        st.subheader("📈 Actual vs Predicted Prices")
        sample_df = listings.copy().sample(100)
        fig, ax = plt.subplots()
        ax.scatter(sample_df['price'], model.predict(sample_df[features]), alpha=0.5)
        ax.plot([0, sample_df['price'].max()], [0, sample_df['price'].max()], 'r--')
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Model Performance: Actual vs Predicted")
        st.pyplot(fig)

# ======= TRAVELER MODE =======
elif section == "🧳 Traveler Mode":
    st.header("🧳 Find Affordable, High-Quality & Equitable Stays")
    st.markdown("""
    Explore listings that are safe, fairly priced, and located in underserved but valuable ZIP codes.  
    Use the filters below to personalize your stay and discover the best options!  
    """)

    underserved = identify_affordable_quality_stays(listings)
    underserved['id'] = underserved['id'].astype(str)

    with st.expander("🎛️ Customize Your Preferences"):
        col1, col2 = st.columns(2)
        with col1:
            max_price = st.slider("💰 Max Price per Person ($)", 10, 300, 100)
            min_beds = st.slider("🛏️ Minimum Beds", 0, 6, 1)
            room_type = st.selectbox("🏡 Room Type", underserved['room_type'].unique())

        with col2:
            zip_code = st.selectbox("📍 Preferred ZIP Code", sorted(underserved['zipcode'].dropna().unique()))
            income_tier = st.radio("💵 Income Tier (Area)", ["Low", "Medium", "High"])
            safety_threshold = st.slider("🛡️ Max Acceptable Crime Rate (if available)", 0.0, 100.0, 50.0)

    # Filter listings
    filtered = underserved[
        (underserved['price_per_person'] <= max_price) &
        (underserved['beds'] >= min_beds) &
        (underserved['room_type'] == room_type) &
        (underserved['zipcode'] == zip_code)
    ]

    if income_tier == "Low":
        filtered = filtered[filtered['INCOME_HOUSEHOLD_MEDIAN'] <= filtered['INCOME_HOUSEHOLD_MEDIAN'].median()]
    elif income_tier == "High":
        filtered = filtered[filtered['INCOME_HOUSEHOLD_MEDIAN'] > filtered['INCOME_HOUSEHOLD_MEDIAN'].median()]

    st.subheader(f"📋 Listings Matching Your Criteria ({len(filtered)})")
    st.dataframe(filtered[['id', 'zipcode', 'price', 'review_scores_rating', 'amenity_count', 'value_score']].head(10))

    st.download_button("⬇️ Download These Listings", filtered.to_csv(index=False), "filtered_travel_listings.csv")

    # Runtime prediction
    st.subheader("🤖 Predict Price for Your Ideal Stay")
    st.markdown("Use the model to estimate price based on your travel preferences.")

    with st.expander("✍️ Enter Stay Details for Price Prediction"):
        col1, col2 = st.columns(2)
    with col1:
        p_accommodates = st.slider("👥 Guests", 1, 10, 2)
        p_bedrooms = st.slider("🛏 Bedrooms", 0, 5, 1)
        p_beds = st.slider("🛌 Beds", 0, 5, 1)
        p_bathrooms = st.slider("🚿 Bathrooms", 0.0, 4.0, 1.0, step=0.5)
        p_amenities = st.slider("📦 Amenities", 1, 30, 10)
        p_room_type = st.selectbox("🏡 Room Type", listings['room_type'].unique(),key="room_type_prediction")
        p_host_ratio = st.slider("📊 Host Listing Ratio", 0.0, 1.0, 0.1)
    with col2:
        p_min_nights = st.slider("📅 Minimum Nights", 1, 30, 2)
        p_max_nights = st.slider("🛎 Maximum Nights", 30, 365, 180)
        p_availability = st.slider("📆 Availability (Days)", 0, 365, 180)
        p_reviews = st.slider("📝 Reviews per Month", 0.0, 10.0, 1.0)
        p_total_reviews = st.slider("⭐ Total Reviews", 0, 500, 20)
        p_verifications = st.slider("🔐 Host Verifications", 0, 5, 2)
        p_zipcode = st.selectbox("📍 ZIP Code", sorted(listings['zipcode'].dropna().unique()),key="zipcode_prediction")

    p_income = listings[listings['zipcode'] == p_zipcode]['INCOME_HOUSEHOLD_MEDIAN'].median()
    price_per_person = listings['price_per_person'].mean()

    # Manual inputs for feature vector
    runtime_input = {
    'accommodates': p_accommodates,
    'bedrooms': p_bedrooms,
    'beds': p_beds,
    'bathrooms': p_bathrooms,
    'amenity_count': p_amenities,
    'reviews_per_month': p_reviews,
    'minimum_nights': p_min_nights,
    'maximum_nights': p_max_nights,
    'availability_365': p_availability,
    'number_of_reviews': p_total_reviews,
    'zipcode': p_zipcode,
    'INCOME_HOUSEHOLD_MEDIAN': p_income,
    'price_per_person': price_per_person,
    'host_listing_ratio': p_host_ratio,
    'is_entire_home': 1 if p_room_type == 'Entire home/apt' else 0,
    'host_verification_count': p_verifications
}


    predict_df = pd.DataFrame([runtime_input])
    predict_df = predict_df.reindex(columns=features, fill_value=0)

    if st.button("📍 Predict Stay Price"):
        pred_price = model.predict(predict_df)[0]
        st.success(f"🎯 Predicted Price for Your Stay: **${pred_price:.2f}**")
        st.markdown("Use this to evaluate if listings in your filtered area are overpriced or great deals!")

    # Top recommended listings
    st.subheader("⭐ Top Recommended Listings (Value + Quality)")

    top_recs = filtered.copy()
    top_recs_features = top_recs.copy()
    top_recs_features = top_recs_features.reindex(columns=features, fill_value=0)

    # Predict
    top_recs['predicted_price'] = model.predict(top_recs_features)

    top_recs['value_score'] = top_recs['predicted_price'] / (top_recs['price'] + 1)
    top_recs['id'] = top_recs['id'].astype(str)
    top_recs = top_recs.sort_values(by='value_score', ascending=False).head(10)

    st.dataframe(top_recs[['id', 'zipcode', 'price', 'predicted_price', 'room_type', 'beds', 'amenity_count', 'review_scores_rating', 'value_score']])


# ======= HOST MODE =======

elif section == "🧑‍💼 Host Mode":
    st.header("🏠 Host Pricing Advisor")
    st.markdown("""
    Optimize your Airbnb pricing strategy using machine learning!  
    Fill in your listing details to get:
    - 📈 Predicted price
    - 💡 Revenue-boosting upgrade suggestions
    - 📊 Comparison with average listings in your ZIP code
    """)

    st.subheader("🔧 Listing Details")

    col1, col2 = st.columns(2)
    with col1:
        accommodates = st.slider("👥 Guests Accommodated", 1, 16, 2)
        bedrooms = st.slider("🏢 Bedrooms", 0, 10, 1)
        beds = st.slider("🏎️ Beds", 0, 10, 1)
        bathrooms = st.slider("🚿 Bathrooms", 0.0, 5.0, 1.0, step=0.5)
        amenity_count = st.slider("📦 Amenities", 1, 50, 10)
        is_entire_home = st.radio("🏡 Entire Home?", ["Yes", "No"]) == "Yes"

    with col2:
        reviews_per_month = st.number_input("📝 Reviews/Month", min_value=0.0, value=1.0)
        number_of_reviews = st.number_input("⭐ Total Reviews", min_value=0, value=10)
        minimum_nights = st.slider("📆 Min Nights Stay", 1, 30, 2)
        maximum_nights = st.slider("🗓️ Max Nights Stay", 30, 365, 180)
        availability_365 = st.slider("📅 Available Days/Year", 0, 365, 180)
        zipcode = st.selectbox("📍 ZIP Code", sorted(listings['zipcode'].dropna().unique()))

    host_listing_ratio = st.slider("📊 Host Listing Ratio", 0.0, 1.0, 0.1)
    host_verification_count = st.slider("🔐 Host Verifications", 0, 5, 2)
    income = st.number_input("💵 Median Income in ZIP", min_value=0, value=60000)
    price_per_person = st.number_input("👤 Price per Person", min_value=0.0, value=30.0)
    offered_price = st.number_input("💵 Your Offered Price ($)", min_value=0.0, value=100.0)

    # --- Build Model Input ---
    user_input = {
        'accommodates': accommodates,
        'bedrooms': bedrooms,
        'beds': beds,
        'bathrooms': bathrooms,
        'amenity_count': amenity_count,
        'reviews_per_month': reviews_per_month,
        'minimum_nights': minimum_nights,
        'maximum_nights': maximum_nights,
        'availability_365': availability_365,
        'number_of_reviews': number_of_reviews,
        'zipcode': zipcode,
        'INCOME_HOUSEHOLD_MEDIAN': income,
        'price_per_person': price_per_person,
        'host_listing_ratio': host_listing_ratio,
        'is_entire_home': 1 if is_entire_home else 0,
        'host_verification_count': host_verification_count
    }

    user_df = pd.DataFrame([user_input])
    user_df_model_input = user_df.reindex(columns=features, fill_value=0)

    if st.button("📊 Predict & Compare"):
        # 🔮 Price Prediction and Recommendations
        predicted_price, recommendations = recommend_host_upgrades_verbose(user_df_model_input.iloc[0], model,offered_price)
        st.success(f"💰 **Predicted Market Price**: ${predicted_price:.2f}")
        st.info(f"🧾 **Your Offered Price**: ${offered_price:.2f}")

        price_diff = predicted_price - offered_price
        if abs(price_diff) < 2:
            st.success("🎯 Your price is well-aligned with market predictions.")
        elif price_diff > 2:
            st.warning(f"💡 You may be undervaluing your listing by ~${price_diff:.2f}.")
        elif price_diff < -2:
            st.warning(f"💡 You may be overvaluing your listing by ~${abs(price_diff):.2f}.")
        # 💡 Upgrade Suggestions
        print(recommendations)
        if len(recommendations)>0:
            st.subheader("💡 Suggested Improvements:")
            for rec in recommendations:
                st.markdown(f"""
                <div style="background-color:#004d00; padding:10px; border-radius:8px; margin-bottom:8px;">
                    <strong>Suggestion:</strong> {rec['suggestion']}  
                    <br><strong>New Price:</strong> ${rec['new_price']:.2f}  
                    <br><strong>Gain:</strong> +${rec['price_gain']:.2f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🎯 Your listing is already well-optimized , No need of improvements in facilities!")
    
        # 📊 ZIP Code Comparison
        st.subheader(f"📍 ZIP Code Comparison: {zipcode}")

        zip_avg = listings[listings['zipcode'] == zipcode][features].mean(numeric_only=True)

        zip_comparison = {
            "Your Listing": user_df_model_input.iloc[0],
            f"ZIP {zipcode} Avg": zip_avg[features]
        }

        comp_df = pd.DataFrame(zip_comparison)
        st.dataframe(comp_df.round(2).T.style.highlight_max(axis=0, color="#006314").highlight_min(axis=0, color="#769F04"))
    

# ======= COMMUNITIES MODE =======
# ======= COMMUNITIES MODE =======
elif section == "🌍 Communities Mode":
    st.header("🌍 Community Equity & Tourism Opportunity Explorer")

    st.markdown("""
    Discover ZIP codes where **equitable tourism** can drive impact.  
    We evaluate areas based on a custom **Equity Score**, combining:
    - 💵 Income levels
    - 💲 Pricing fairness
    - ⭐ Listing quality  
    Explore how tourism can uplift underserved neighborhoods by identifying **high-equity zones** worth promoting.
    """)

    st.divider()

    # --- Compute ZIP stats and equity listings
    zip_stats, equity_listings = compute_zipcode_equity_score(listings, listings)

    # --- Toast success message
    st.toast("✅ Equity ZIPs loaded! Explore high-impact neighborhoods ✨")

    # --- Summary Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("📍 ZIP Codes Evaluated", len(zip_stats))
    col2.metric("⚖️ Avg Equity Score", f"{zip_stats['equity_score'].mean():.2f}")
    col3.metric("🏡 Listings in High-Equity ZIPs", len(equity_listings))

    st.subheader("🏅 Top 10 ZIP Codes by Equity Score")

    top_zip_scores = zip_stats.sort_values(by="equity_score", ascending=False).head(10)

    # --- Horizontal Bar Chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_zip_scores['zipcode'].astype(str), top_zip_scores['equity_score'], color='#2a9d8f')
    ax.set_xlabel("Equity Score")
    ax.set_ylabel("ZIP Code")
    ax.set_title("Top ZIP Codes by Equity Score")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', fontsize=9)

    st.pyplot(fig)

    st.divider()
    print(top_zip_scores.head)
    # --- Listings Table
    st.subheader("🏘️ Listings in High-Equity ZIP Codes")

    equity_listings['id'] = equity_listings['id'].astype(str)
    equity_listings_display = equity_listings[['id', 'zipcode', 'price', 'review_scores_rating', 'room_type', 'amenity_count']]
    equity_listings_display = equity_listings_display.sort_values(by='review_scores_rating', ascending=False)

    st.dataframe(equity_listings_display.head(20), use_container_width=True)

    st.download_button("⬇️ Download Equity Listings", equity_listings_display.to_csv(index=False), "equity_listings.csv")

    st.divider()

    # --- Explanation
    with st.expander("📚 How is the Equity Score Calculated?"):
        st.markdown("""
        The **Equity Score** is a composite metric that identifies ZIP codes offering both affordability and quality, particularly in lower-income neighborhoods.  

        **Calculation Factors**:
        - 💵 **Income Score** → Lower median income = higher equity need (weighted 40%)
        - 💲 **Price Score** → Lower average price = better affordability (weighted 30%)
        - ⭐ **Quality Score** → Higher listing ratings = better experience (weighted 30%)

        The final score helps promote listings in areas where tourism can support local economic development without compromising traveler experience.
        """)

    st.markdown("---")


# ======= CONCLUSION =======
elif section == "📌 Conclusion":
    st.header("Key Takeaways")
    st.markdown("""
    ✅ Hosts can increase their price through small upgrades like amenities or room type.
    ✅ Travelers can find hidden gems in affordable ZIP codes.
    ✅ Community equity can be improved by promoting quality listings in overlooked neighborhoods.
    """)
    st.balloons()
    with st.popover("🎁 THANK YOU FOR VISITING US!"):
        st.markdown("""
        We hope our insights helped you improve your Airbnb strategy!  
        If you enjoyed this tool, consider sharing it with your friends or giving feedback. 💬  
        
        🔗 [Follow us on LinkedIn](https://www.linkedin.com/in/muhammad-tayyab-42792a262/)  
        💌 [Contact Support](mailto:tyb3122@gmail.com)  
        🙌 Stay safe and host smart!
        ---
        👨‍💻 Developed by [**Muhammad Tayyab**](https://github.com/tyb01)  
        ⭐ Feel free to explore the code and give it a star!
        """)
    
