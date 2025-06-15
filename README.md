# Airbnb Equity & Pricing Intelligence Dashboard

A Streamlit-based interactive web app to help:
- 🧑‍💼 **Hosts** get pricing and amenity recommendations
- 🧳 **Travelers** find affordable, high-quality stays
- 🌍 **Communities** promote equitable tourism across ZIP codes

---

## 📁 Project Structure
```bash
├── app.py                     # Streamlit app
├── requirements.txt           # Dependencies
├── README.md   
|__ ALL FILES CREATED FOR UI DISPLAY.        # Documents
├── data/                      # Raw and cleaned data
│   ├── listings.csv
│   └── IncomeHouseholdMedian.xlsx
├── outputs/                   # SHAP plots, model outputs
│   ├── shap_summary_bar.png   # in next version
│   └── shap_summary_full.png  # in next version
└── scripts/                   # Modular logic
    ├── data.py
    ├── features.py
    ├── model.py
    ├── recommend.py
    ├── equity.py

```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/tyb01/airbnb-equity-analyzer
cd airbnb-equity-analyzer
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

##  Core Features

###  Introduction
Overview of the project goals and societal impact.

###  EDA
Basic visualizations to explore price trends, room types, and distributions.

###  Model
Train and evaluate a Random Forest regressor to predict listing prices.
Detect undervalued listings.

###  Traveler Mode
- Filter listings by price, review score, and ZIP equity.
- View high-value listings in underserved areas.

###  Host Mode
- Input listing attributes to get predicted price.
- View suggestions for amenity or room-type upgrades.

###  Communities Mode
- Score ZIP codes based on income, price, and quality.
- Identify equitable tourism zones.

---

##  Data Sources
- `listings.csv`: Airbnb listing data.
- `IncomeHouseholdMedian.xlsx`:ZIP code median incomes.

---

## ✅ Outputs
- `undervalued_listings.csv`
- `zip_equity_scores.csv`
- `recommended_listings.pkl`
- `equity_recommended_listings.csv`
- `shap_summary_bar.png`

---

## 👩‍💻 Contributions
Pull requests are welcome! Feel free to fork the repo and improve modularity, UI, or performance.

---

## 📃 License
MIT License. Free to use and modify with attribution.

---

## 🙌 Acknowledgments
Thanks to the Airbnb community datasets and the idea of fair and equitable data science in real estate!
