"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    # 12-hour mean per patient, and mean per hour column
    patient_means = readings.mean(axis=1)
    hour_means = readings.mean(axis=0)

    # Highest patient and peak hour
    top = patient_means.argmax()
    peak = hour_means.argmax()

    # Monitor averages, in sorted(set(monitors)) order
    monitor_ids = sorted(set(monitors))
    monitor_avgs = []
    for m in monitor_ids:
        mask = monitors == m
        monitor_avgs.append(patient_means[mask].mean())
    monitor_avgs = np.array(monitor_avgs)

    high_monitor = monitor_ids[monitor_avgs.argmax()]
    high_mask = monitors == high_monitor
    monitor_offset = patient_means[high_mask].mean() - patient_means[~high_mask].mean()

    results = {
        "patients": readings.shape[0],
        "readings": readings.size,
        "mean_sbp": round(float(readings.mean()), 2),
        "sd_sbp": round(float(readings.std()), 2),
        "min_sbp": int(readings.min()),
        "max_sbp": int(readings.max()),
        "stage2_patients": int((patient_means >= 140).sum()),
        "highest_patient": str(patient_ids[top]),
        "highest_patient_mean": round(float(patient_means[top]), 2),
        "peak_hour_column": str(hour_columns[peak]),
        "peak_hour_mean": round(float(hour_means[peak]), 2),
        "high_monitor": str(high_monitor),
        "monitor_offset": round(float(monitor_offset), 2),
        "stage2_other_monitors": int((patient_means[~high_mask] >= 140).sum()),
    }

    with open("output/vitals_summary.txt", "w") as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
