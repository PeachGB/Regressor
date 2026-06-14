"""End-to-end pipeline: pandas in, encode labels, scale features, classify.

Shows the pieces working together on a small train/test split:
  LabelEncoder (string target -> 0/1) + StandardScaler (features) + LogisticRegression.

Run with:  python regressor-python/examples/05_pipeline.py
"""
import numpy as np
import pandas as pd

from regressor.preprocessing import StandardScaler, LabelEncoder
from regressor.model.linear_model import LogisticRegression, Penalty

# --- A tiny labelled dataset as a pandas DataFrame --------------------------
# Two species separable by petal length / width.
rng = np.random.default_rng(123)
setosa = pd.DataFrame(
    {
        "petal_length": rng.normal(1.5, 0.2, 50),
        "petal_width": rng.normal(0.3, 0.1, 50),
        "species": "setosa",
    }
)
versicolor = pd.DataFrame(
    {
        "petal_length": rng.normal(4.5, 0.4, 50),
        "petal_width": rng.normal(1.3, 0.2, 50),
        "species": "versicolor",
    }
)
data = pd.concat([setosa, versicolor], ignore_index=True)
data = data.sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle

# --- Train / test split -----------------------------------------------------
split = int(0.75 * len(data))
train, test = data.iloc[:split], data.iloc[split:]

feature_cols = ["petal_length", "petal_width"]

# --- Encode the string target into 0/1 --------------------------------------
encoder = LabelEncoder()
y_train = encoder.fit_transform(train["species"].tolist())
y_test = encoder.transform(test["species"].tolist())
print("Label encoding:", {name: i for i, name in enumerate(encoder.classes())})

# --- Standardize features (fit on train, apply to both) ---------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_test = scaler.transform(test[feature_cols])

# --- Fit the classifier -----------------------------------------------------
clf = LogisticRegression(learning_rate=0.3, epochs=2000)
clf.set_penalty(Penalty.l2(0.05))
clf.fit(X_train, y_train)

train_acc = clf.score(y_train, clf.predict(X_train))
test_acc = clf.score(y_test, clf.predict(X_test))
print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test  accuracy: {test_acc:.3f}")

# --- Classify a couple of brand-new flowers ---------------------------------
samples = pd.DataFrame(
    {"petal_length": [1.4, 4.7], "petal_width": [0.2, 1.4]}
)
probs = clf.predict(scaler.transform(samples))
classes = encoder.classes()
print("\nNew predictions:")
for (_, row), p in zip(samples.iterrows(), probs):
    predicted = classes[1] if p >= 0.5 else classes[0]
    print(
        f"  petal {row.petal_length}x{row.petal_width} -> "
        f"P({classes[1]})={p:.3f} -> {predicted}"
    )
