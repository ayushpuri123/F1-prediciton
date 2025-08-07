import numpy as np
import pandas as pd
from model_training import model_balanced, X_test, features_df
from data_loader import load_datasets
data = load_datasets() 

def simulate_race(probabilities, drivers, n_simulations=10000):
    simulations = np.random.choice(drivers, size=n_simulations, p=probabilities)
    win_counts = pd.Series(simulations).value_counts(normalize=True) * 100
    return win_counts

# Predict win probabilities using the best model
probas = model_balanced.predict_proba(X_test)[:, 1]

# Create a copy of X_test and attach probabilities + driverId
X_test_sim = X_test.copy()
X_test_sim['proba'] = probas
X_test_sim['driverId'] = features_df.loc[X_test.index, 'driverId'].values

# Aggregate average win probability per driver
X_test_sim = X_test_sim.groupby('driverId')['proba'].mean().reset_index()
X_test_sim['proba'] /= X_test_sim['proba'].sum()

# Run Monte Carlo simulation
simulated_results = simulate_race(X_test_sim['proba'].values, X_test_sim['driverId'].values)
# Reset index to make driverId a column
simulated_results = simulated_results.reset_index()
simulated_results.columns = ['driverId', 'win_pct']

# Merge with driver names
driver_info = data['drivers'][['driverId', 'surname', 'forename']]
simulated_results = simulated_results.merge(driver_info, on='driverId', how='left')

# Sort and display top drivers by win percentage
simulated_results = simulated_results.sort_values(by='win_pct', ascending=False)
print(simulated_results[['driverId', 'forename', 'surname', 'win_pct']].head(30).round(2))