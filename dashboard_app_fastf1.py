import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import fastf1
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime, timedelta

# Set up Streamlit page
st.set_page_config(page_title="F1 Race Winner Predictor", layout="wide")

# Sidebar Setup
st.sidebar.title("Race Selection")
year = st.sidebar.selectbox("Select Year", options=[2022, 2023, 2024, 2025], index=2)
all_races = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Emilia Romagna (Imola)", "Monaco", "Spain", "Canada", "Austria", "Britain",
    "Hungary", "Belgium", "Netherlands", "Italy (Monza)", "Singapore",
    "USA (Austin)", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]
race_name = st.sidebar.selectbox("Select Grand Prix", all_races)

# Countdown Timer Example (for upcoming race)
st.sidebar.markdown("---")
st.sidebar.header("⏳ Countdown to Race")
next_race_date = datetime(2025, 4, 27, 23, 0, 0)  # Example: 27 April 2025, 11 PM
now = datetime.now()
time_left = next_race_date - now
if time_left.total_seconds() > 0:
    st.sidebar.success(f"{time_left.days}d {time_left.seconds//3600}h {(time_left.seconds//60)%60}m remaining")
else:
    st.sidebar.warning("Race has started or finished!")

# Main Page Setup
st.markdown("""
# 🏎️ F1 Race Winner Predictor (Live FastF1)
Welcome to the F1 live race prediction dashboard! This tool uses real-time qualifying data and machine learning to predict race winners.

---

## 📅 Overview
- Real-time qualifying lap time integration
- Machine learning-based win probability prediction
- Simulated race outcomes (Monte Carlo)
- Driver performance visualization

---
""")

# Enable FastF1 cache
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

# Load Session
try:
    session = fastf1.get_session(year, race_name, 'Qualifying')
    session.load()
except Exception as e:
    st.error(f"Error loading session: {e}")
    st.stop()

# Feature Engineering
laps = session.laps.pick_quicklaps()
drivers = laps['Driver'].unique()

best_laps_list = []
for driver in drivers:
    driver_laps = laps[laps['Driver'] == driver]
    if not driver_laps.empty:
        best_lap = driver_laps.pick_fastest()
        best_laps_list.append({
            'Driver': best_lap['Driver'],
            'LapTime': best_lap['LapTime']
        })

best_laps = pd.DataFrame(best_laps_list)

features = best_laps.copy()
features['BestLapTime'] = features['LapTime'].apply(lambda x: x.total_seconds())

quali_results = session.results
features = features.merge(quali_results[['Abbreviation', 'Position']], left_on='Driver', right_on='Abbreviation', how='left')

features['grid_position'] = features['Position']
features['driver_form'] = 1 / (features['BestLapTime'] + 1e-5)

X_live = features[['grid_position', 'driver_form']]

# Load Model
try:
    model = joblib.load('f1_random_forest_model.pkl')
    st.success("✅ Loaded pre-trained model.")
except:
    st.warning("⚠️ Pre-trained model not found. Training a new model from current session data...")
    features['target'] = (features['grid_position'] <= 3).astype(int)
    X_temp = features[['grid_position', 'driver_form']]
    y_temp = features['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_temp, y_temp)
    joblib.dump(model, 'f1_random_forest_model.pkl')
    st.success("✅ Trained and saved new model based on current session.")

# Predict Win Probabilities
win_probs = model.predict_proba(X_live)[:, 1]
features['Predicted Win Probability'] = win_probs
features = features.sort_values(by='Predicted Win Probability', ascending=False)
features['Predicted Win Probability'] = features['Predicted Win Probability'] / features['Predicted Win Probability'].sum()

# Top Predicted Winner Card
st.markdown("""
<div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px;'>
<h2 style='text-align: center;'>🏆 Top Predicted Winner</h2>
<h3 style='text-align: center;'>{} ({:.2f}% win chance)</h3>
</div>
""".format(features.iloc[0]['Driver'], features.iloc[0]['Predicted Win Probability'] * 100), unsafe_allow_html=True)

st.markdown("---")

# Monte Carlo Simulation
st.subheader("📊 Simulated Win Probabilities (10,000 Races)")

def simulate_race(probabilities, drivers, n_simulations=10000):
    sims = np.random.choice(drivers, size=n_simulations, p=probabilities)
    win_counts = pd.Series(sims).value_counts(normalize=True) * 100
    return win_counts

sim_results = simulate_race(features['Predicted Win Probability'].values, features['Driver'].values)
top_sim_results = sim_results.sort_values(ascending=False).head(15).reset_index()
top_sim_results.columns = ['Driver', 'win_pct']

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=top_sim_results, x='win_pct', y='Driver', palette='Blues_d', ax=ax)
plt.xlabel('% of Simulated Wins')
plt.ylabel('Driver')
plt.title('Top 15 Predicted Winners from 10,000 Simulations')
st.pyplot(fig)

st.markdown("---")

# Driver Probabilities Table
st.subheader("🔢 Driver Probabilities Table")
st.dataframe(features[['Driver', 'Predicted Win Probability']])

csv = features.to_csv(index=False).encode('utf-8')
st.download_button("Download Win Probabilities", csv, "f1_live_predictions.csv", "text/csv")

st.markdown("---")

# Model Evaluation after Race
st.subheader("🔄 Model Evaluation After Race")
if st.button("Pull Race Results and Evaluate"):
    try:
        race_session = fastf1.get_session(year, race_name, 'Race')
        race_session.load()
        race_results = race_session.results

        actual_winner = race_results[race_results['Position'] == 1]['Abbreviation'].values[0]
        predicted_winner = features.iloc[0]['Driver']

        st.success(f"✅ Actual Winner: {actual_winner}")
        st.success(f"✅ Predicted Winner: {predicted_winner}")

        if actual_winner == predicted_winner:
            st.success("🎯 Correct Prediction!")
        else:
            st.error("❌ Wrong Prediction!")
    except Exception as e:
        st.error(f"Error loading race results: {e}")

st.markdown("---")

# Feature Importance Chart
st.subheader("🔍 Feature Importance")
try:
    importances = model.feature_importances_
    feature_names = X_live.columns
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"))
except Exception as e:
    st.error(f"Error displaying feature importance: {e}")
