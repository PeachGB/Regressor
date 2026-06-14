"""Preprocessing utilities: StandardScaler, MinMaxScaler, LabelEncoder.

Run with:  python regressor-python/examples/04_preprocessing.py
"""
import numpy as np

from regressor.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Features on wildly different scales (e.g. age in years, income in dollars).
raw = np.array(
    [
        [25.0, 30_000.0],
        [40.0, 60_000.0],
        [60.0, 90_000.0],
        [35.0, 45_000.0],
    ]
)

print("Raw features:\n", raw)

# --- StandardScaler: zero mean, unit (sample) std ---------------------------
scaler = StandardScaler()
standardized = scaler.fit_transform(raw)
print("\nStandardScaler -> zero mean, unit std (ddof=1):")
print(np.round(standardized, 3))
print("  column means:", np.round(standardized.mean(axis=0), 6))
print("  column stds :", np.round(standardized.std(axis=0, ddof=1), 6))

# fit once, reuse on new rows
new_rows = np.array([[50.0, 75_000.0]])
print("\nTransform an unseen row with the learned statistics:")
print("  ", np.round(scaler.transform(new_rows), 3))

# --- MinMaxScaler: rescale each column into [0, 1] --------------------------
mm = MinMaxScaler().fit_transform(raw)
print("\nMinMaxScaler -> each column in [0, 1]:")
print(np.round(mm, 3))

# --- LabelEncoder: categories -> integer codes ------------------------------
labels = ["dog", "cat", "bird", "cat", "dog"]
enc = LabelEncoder()
codes = enc.fit_transform(labels)
print("\nLabelEncoder:")
print("  classes (sorted):", enc.classes())
print("  labels :", labels)
print("  codes  :", codes)
print("  inverse:", enc.inverse_transform(codes))
