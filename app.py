import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
PREP_PATH = os.path.join(BASE, "prep.pkl")          # preprocessing (scaler + encoder)
XGB_PATH = os.path.join(BASE, "xgb_model.json")     # tuned XGBoost model
MAE_XGB = 1.739                                     # from test set evaluation

RATECODES = {
    1: "Standard rate",
    2: "JFK (flat fare)",
    3: "Newark",
    4: "Nassau / Westchester",
    5: "Negotiated fare",
    6: "Group ride",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_model():
    from xgboost import XGBRegressor
    prep = joblib.load(PREP_PATH)
    model = XGBRegressor()
    model.load_model(XGB_PATH)
    return prep, model


st.title("NYC Taxi Fare Predictor")
st.caption("Estimate the taxi fare before the trip starts.")

tab_pred, tab_perf, tab_insight = st.tabs(["Predict", "Model performance", "Business insights"])

with tab_pred:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Trip details")
        distance = st.number_input("Trip distance (miles)", 0.1, 100.0, 3.0, 0.1)
        passengers = st.number_input("Passengers", 1, 6, 1, 1)
        hour = st.number_input("Pickup hour (0-23)", 0, 23, 12, 1)
        day = st.selectbox("Day of week", DAYS, index=2)
        rate = st.selectbox("Rate type", list(RATECODES), format_func=lambda k: f"{k} - {RATECODES[k]}")
        pu = st.number_input("Pickup zone ID (1-265)", 1, 265, 161)
        do = st.number_input("Dropoff zone ID (1-265)", 1, 265, 236)
        go = st.button("Predict fare", type="primary")

    with right:
        st.subheader("Prediction")
        if go:
            X = pd.DataFrame([{
                "trip_distance": distance,
                "passenger_count": int(passengers),
                "pickup_hour": int(hour),
                "PULocationID": int(pu),
                "DOLocationID": int(do),
                "day_of_week": day,
                "RatecodeID": int(rate),
                "is_weekend": int(day in ("Saturday", "Sunday")),
            }])
            try:
                prep, model = load_model()
                pred = float(model.predict(prep.transform(X))[0])
                st.metric("Estimated fare", f"${pred:,.2f}")
                st.write(f"Typical error: about ±${MAE_XGB:.2f}  →  range ${max(pred - MAE_XGB, 0):,.2f} to ${pred + MAE_XGB:,.2f}")
                if int(rate) == 2:
                    st.info("JFK trips use a flat fare, so distance matters less here.")
            except Exception as e:
                st.error("The model could not be loaded or used. Check that prep.pkl and xgb_model.json are in the same folder as app.py.")
                st.code(f"{type(e).__name__}: {e}")
        else:
            st.info("Fill in the trip details and click **Predict fare**.")

with tab_perf:
    st.subheader("Test-set results (80/20 split)")
    perf = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "Neural Network (Keras)", "XGBoost (tuned)"],
        "MAE ($)": [2.498, 1.852, 1.762, 1.964, 1.739],
        "RMSE ($)": [4.159, 3.282, 3.228, 3.344, 3.195],
        "R²": [0.923, 0.952, 0.954, 0.951, 0.955],
    })
    st.dataframe(perf, hide_index=True, use_container_width=True)
    st.bar_chart(perf.set_index("Model")["MAE ($)"])
    st.write("5-fold cross-validation R² (XGBoost): **0.9538** (folds range 0.945 - 0.960), so the model is stable and not overfitting.")

with tab_insight:
    st.subheader("What the data tells us")
    st.markdown(
        """
- **Distance is the main driver** of the fare (correlation 0.95).
- **JFK flat fare (~$70)** creates a flat line in the distance-vs-fare chart, so those trips do not follow the usual per-mile pattern.
- **Average fare peaks around 5 AM** (~$25), likely airport trips in the early morning.
- **Day of week and passenger count barely change the fare.**
- `duration_min` and `speed_mph` were **not used** as features because they are only known after the trip ends (data leakage).
        """
    )
    st.subheader("Recommendations")
    st.markdown(
        """
1. Use the model for **upfront fare estimates** shown to passengers before the ride.
2. **Plan airport coverage in the early morning** (around 5 AM), when average fares are highest.
3. Treat **airport flat-fare trips as a separate segment** in pricing and revenue forecasts.
4. **Retrain the model regularly** (e.g. monthly) and monitor MAE to detect drift.
        """
    )
