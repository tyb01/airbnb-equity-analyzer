# Or apply to the whole dataset (⚠️ very slow, better to use caching or batches)
# zip_codes = []
# for idx, row in listings.iterrows():
#     lat, lon = row['latitude'], row['longitude']
#     zip_code = get_zip_code(lat, lon)
#     zip_codes.append(zip_code)
#     print(f"Processed index {idx}, ZIP: {zip_code}")
    
# # Add column
# listings['zipcode'] = zip_codes