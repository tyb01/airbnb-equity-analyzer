# scripts/recommend.py

import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_with_shap(model, X_train, save_dir="outputs"):
    """
    Generate SHAP summary plots and save to disk.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
    plt.savefig(f"{save_dir}/shap_summary_bar.png", bbox_inches='tight')

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"{save_dir}/shap_summary_full.png", bbox_inches='tight')

    print("✅ SHAP plots saved.")
    return shap_values

def recommend_host_upgrades_verbose(row, model,user_offered_price):
    """
    Suggest improvements (add amenities, switch room type) and compute price gain.
    """
    row = row.copy()
    original_price = model.predict([row])[0]
    recommendations = []

    #IF UNDERVALUED
    gain = original_price - user_offered_price
    if gain > 0:
         recommendations.append({
                'feature': 'undervalued',
                'suggestion': 'Just increase your price listing to new price without refinments in property.',
                'current_price': round(user_offered_price, 2),
                'new_price': round(original_price, 2),
                'price_gain': round(gain, 2)
            })
    # Simulate: Add amenities
    if 'amenity_count' in row:
        mod = row.copy()
        mod['amenity_count'] += 5
        new_price = model.predict([mod])[0]
        gain = new_price - original_price
        if gain > 0:
            recommendations.append({
                'feature': 'amenity_count',
                'suggestion': 'Add 5 more amenities',
                'current_price': round(original_price, 2),
                'new_price': round(new_price, 2),
                'price_gain': round(gain, 2)
            })

    # Simulate: Switch to entire home
    if 'room_type_Private room' in row and row['room_type_Private room'] == 1:
        mod = row.copy()
        mod['room_type_Private room'] = 0
        if 'room_type_Entire home/apt' in row:
            mod['room_type_Entire home/apt'] = 1
        new_price = model.predict([mod])[0]
        gain = new_price - original_price
        if gain > 0:
            recommendations.append({
                'feature': 'room_type',
                'suggestion': 'Switch to Entire Home/Apt',
                'current_price': round(original_price, 2),
                'new_price': round(new_price, 2),
                'price_gain': round(gain, 2)
            })
    print(recommendations)
    return round(original_price, 2), recommendations
