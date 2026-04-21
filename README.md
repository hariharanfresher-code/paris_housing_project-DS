# 🏙️ Paris Housing Prices — Data Science Project

A complete end-to-end data science project analysing and modelling residential property prices across all 20 Paris arrondissements.

---

## 📁 Project Structure

```
paris_housing_ds/
├── data/
│   └── paris_housing_prices_dataset.csv   # Raw dataset (1 200 properties)
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb        # Feature Engineering & Preprocessing
│   └── 03_modelling.ipynb                  # Model Training, Evaluation & Prediction
├── src/
│   └── utils.py                            # Shared helper functions
├── outputs/                                # Saved plots & model artefacts
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Column | Type | Description |
|---|---|---|
| `Property_ID` | str | Unique property identifier |
| `Arrondissement` | int | Paris district (1–20) |
| `Property_Type` | str | Apartment / Studio / Loft / Penthouse |
| `Size_sqm` | int | Property size in m² |
| `Rooms` | int | Number of rooms |
| `Floor` | int | Floor number |
| `Year_Built` | int | Year the property was built |
| `Condition` | str | New / Renovated / Good / Needs Renovation |
| `Distance_to_Center_km` | float | Distance to Paris city centre (km) |
| `Price_EUR` | float | **Target** — Sale price in Euros |

**1 200 rows · 10 columns · No missing values**

---

## 🚀 Quick Start

```bash
# 1. Clone / create the repo
git init paris_housing_ds && cd paris_housing_ds

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter lab notebooks/
```

Run notebooks in order: **01 → 02 → 03**

---

## 🔬 What Each Notebook Does

### 01 · Exploratory Data Analysis
- Distribution of prices, size, rooms, floor, year built
- Price variation by arrondissement (choropleth-style bar chart)
- Correlation heatmap
- Box plots — price by property type & condition
- Scatter: price vs. size, price vs. distance to centre

### 02 · Feature Engineering
- Label encoding of categorical features
- New feature: `price_per_sqm`
- New feature: `property_age`
- New feature: `is_central` (arrondissement ≤ 8)
- Train / test split (80 / 20)
- StandardScaler normalisation

### 03 · Modelling
- Baseline: Linear Regression
- Random Forest Regressor (tuned)
- XGBoost Regressor (tuned)
- Evaluation: RMSE, MAE, R²
- Feature importance chart
- Residual analysis
- Predict price for a new property

---

## 📦 Requirements

See `requirements.txt`. Key libraries:

- `pandas`, `numpy` — data manipulation  
- `matplotlib`, `seaborn` — visualisation  
- `scikit-learn` — preprocessing & modelling  
- `xgboost` — gradient boosting  
- `jupyter` / `jupyterlab` — notebook environment  

---

## 📈 Results (example)

| Model | RMSE (€) | MAE (€) | R² |
|---|---|---|---|
| Linear Regression | ~320 000 | ~240 000 | ~0.62 |
| Random Forest | ~180 000 | ~130 000 | ~0.84 |
| XGBoost | ~165 000 | ~118 000 | ~0.87 |

*(Actual results will vary depending on random seed.)*

---

## 🤝 Contributing

Pull requests welcome. Please open an issue first for major changes.

## 📄 Licence

MIT
