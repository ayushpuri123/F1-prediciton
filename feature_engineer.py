import pandas as pd
def compute_driver_form(results_df, window=5):
    results_df = results_df.sort_values(by=['driverId', 'raceId'])
    results_df['positionOrder'] = pd.to_numeric(results_df['positionOrder'], errors='coerce')
    results_df['driver_form'] = results_df.groupby('driverId')['positionOrder'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return results_df

def compute_track_history(results_df):
    track_performance = results_df.groupby(['driverId', 'circuitId'])['positionOrder'].mean().reset_index()
    track_performance.rename(columns={'positionOrder': 'track_history_avg'}, inplace=True)
    results_df = results_df.merge(track_performance, on=['driverId', 'circuitId'], how='left')
    return results_df

def compute_team_strength(results_df):
    team_strength = results_df.groupby(['constructorId'])['points'].mean().reset_index()
    team_strength.rename(columns={'points': 'constructor_avg_points'}, inplace=True)
    results_df = results_df.merge(team_strength, on='constructorId', how='left')
    return results_df

def compute_dnf_rate(results_df):
    results_df['did_finish'] = results_df['statusId'].apply(lambda x: 1 if x == 1 else 0)
    dnf_rate = results_df.groupby('driverId')['did_finish'].mean().reset_index()
    dnf_rate.rename(columns={'did_finish': 'finish_rate'}, inplace=True)
    results_df = results_df.merge(dnf_rate, on='driverId', how='left')
    return results_df

def engineer_features(results_df):
    df = compute_driver_form(results_df)
    df = compute_track_history(df)
    df = compute_team_strength(df)
    df = compute_dnf_rate(df)
    return df