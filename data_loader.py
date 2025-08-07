import pandas as pd
import os

def load_datasets():
    # Automatically detects the current script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "DataSets")

    datasets = {}
    files = [
        "circuits.csv", "constructor_results.csv", "constructor_standings.csv",
        "constructors.csv", "driver_standings.csv", "drivers.csv", "lap_times.csv",
        "pit_stops.csv", "qualifying.csv", "races.csv", "results.csv",
        "seasons.csv", "sprint_results.csv", "status.csv"
    ]
    for file in files:
        full_path = os.path.join(data_path, file)
        datasets[file.replace(".csv", "")] = pd.read_csv(full_path)
    return datasets

if __name__ == "__main__":
    data = load_datasets()
    for name, df in data.items():
        print(f"{name}: {df.shape}")