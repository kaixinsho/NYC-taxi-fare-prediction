# NYC Taxi Fare Prediction

A machine learning project that predicts the **base fare of a New York City yellow taxi trip before the trip starts**, using distance, time, day, rate type and pickup/dropoff zone. The best model is shown in an interactive **Streamlit** web app.

Built as the take home test for the Data Science & Machine Learning program at Dibimbing (October 2026).

## Problem Statement

A taxi fare is only known after the trip ends. If it can be estimated upfront, operators can plan pricing and estimate revenue, and passengers get a fair expectation of the price.

> How can we predict the NYC taxi fare before the trip starts, using distance, time, day and zone, so operators can plan pricing and estimate revenue?

## Dataset

- **Source:** Kaggle, *Cleaned NYC Taxi Trip Data (2025 Sample)* by Aryan Patel (Apache 2.0), built from NYC TLC yellow taxi records.
- **Structure:** 25 columns (trip, time, payment and fee information).
- **Used in this project:** the first 200,000 rows of the file (to keep the notebook fast). The time coverage of this sample has not been checked, so it may not cover the whole year.
- **Target:** `fare_amount` (base fare in USD).
- The CSV is large (about 600 MB) and is **not included** in this repository. Download it from Kaggle if you want to re-run the notebook.
- Note: the dataset page shows a row count of 3,250,000 with a "Replace with your number" remark. That looks like an unfilled template, so the shape was verified in code instead.

## Project Structure

| File | Purpose |
|---|---|
| Notebook (`.ipynb`) | Full analysis: cleaning, EDA, modelling, tuning |
| `app.py` | Streamlit web app |
| `xgb_model.json` | Trained (tuned) XGBoost model, saved in XGBoost's native format |
| `prep.pkl` | Fitted preprocessing (scaler + one-hot encoder) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container setup for the app (not yet tested) |

## Methodology

**1. Cleaning**
- Dropped duplicate rows (none found).
- `Airport_fee` was the only column with missing values (24,951 of 200,000). Missing means no airport fee, so it was filled with 0.
- Removed invalid rows: fare ≤ 0 or ≥ 300, distance ≤ 0 or ≥ 100 miles, passengers outside 1-6.
- Result: **198,578 trips** (about 0.7% removed).

**2. Features (no data leakage)**
Only information known **before** the trip starts is used:
`trip_distance`, `passenger_count`, `pickup_hour`, `PULocationID`, `DOLocationID`, `day_of_week`, `RatecodeID`, `is_weekend`.

Columns such as `total_amount`, `tip_amount`, `duration_min`, `speed_mph`, taxes and surcharges are **excluded**, because they are only known after the trip or already contain the fare.

**3. Preprocessing pipeline**
`StandardScaler` for numeric columns and `OneHotEncoder` for `day_of_week` and `RatecodeID`, wrapped in a scikit-learn `Pipeline` with the model. It is fitted on the training set only (80/20 split, `random_state=42`).

**4. Models**
- Linear Regression (baseline)
- Random Forest
- XGBoost
- Neural network (TensorFlow/Keras): Dense 128 → Dropout 0.1 → Dense 64 → Dense 32 → Dense 1, ReLU, Adam (lr 0.001), batch size 512, early stopping (stopped at epoch 18)

**5. Optimization**
- `RandomizedSearchCV` on XGBoost (8 combinations, 3-fold, scored by MAE, on a 50,000-row subset).
- Best parameters: `n_estimators=200`, `max_depth=8`, `learning_rate=0.1`, `subsample=1.0`.
- 5-fold cross-validation on XGBoost (50,000-row subset, before tuning): mean R² = **0.9538** (folds 0.945 to 0.960).

## Results

Test set (20% hold-out):

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 2.498 | 4.159 | 0.923 |
| Random Forest | 1.852 | 3.282 | 0.952 |
| XGBoost | 1.762 | 3.228 | 0.954 |
| Neural Network (Keras) | 1.964 | 3.344 | 0.951 |
| **XGBoost (tuned)** | **1.739** | **3.195** | **0.955** |

The tuned XGBoost is off by about **$1.74** on average. The high R² is expected because taxi fares are calculated by a meter, mostly from distance and time.

## Key Insights

- **Distance is the main driver** of the fare (correlation 0.95).
- A flat line at about **$70** in the distance-vs-fare chart is most likely the JFK airport flat fare (`RatecodeID` = 2), so those trips do not follow the per-mile pattern.
- The average fare peaks around **5 AM** (about $25), most likely because of airport trips.
- **Day of week and passenger count** barely change the fare.
- Typical fare: median $12.80, mean $17.63.

## Web App (Streamlit)

The app has three tabs: **Predict** (enter trip details, get the estimated fare and an error range of about ±$1.74), **Model performance**, and **Business insights**.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `prep.pkl` and `xgb_model.json` in the same folder. Use an XGBoost version equal to or newer than the one used for training. Models saved by a different XGBoost version may fail to load.

### Docker (optional)

```bash
docker build -t nyc-taxi-fare .
docker run -p 8501:8501 nyc-taxi-fare
```

The Dockerfile has been written but not yet tested.

## Limitations

- Trained on a 200,000-row sample whose time coverage has not been checked.
- Zone IDs are treated as numbers, but they are really categories.
- Predictions are for the base fare only (no tolls, tips or surcharges).
- No monitoring or automatic retraining yet.

## Next Steps

- Sample randomly across the whole year and re-train.
- Encode pickup/dropoff zones properly (for example target encoding).
- Add weather and event data.
- Deploy the app online and add model monitoring.

## Author

Gabriella Annabelle Antoniette
