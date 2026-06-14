"""Binary classification with logistic regression.

A toy "did the student pass?" problem: the more hours studied, the more likely a
pass. `predict` returns class probabilities; threshold at 0.5 for a label, and
`score` reports accuracy.

Run with:  python regressor-python/examples/03_logistic_regression.py
"""
import numpy as np

from regressor.model.linear_model import LogisticRegression, Penalty

# --- Synthetic, linearly-separable-ish data ---------------------------------
rng = np.random.default_rng(7)
hours = np.concatenate([rng.uniform(0, 3, 40), rng.uniform(4, 8, 40)])
passed = (hours > 3.5).astype(float)
order = rng.permutation(len(hours))
hours, passed = hours[order], passed[order]

X = hours.reshape(-1, 1)
y = passed

# --- Fit with a small L2 penalty --------------------------------------------
clf = LogisticRegression(learning_rate=0.5, epochs=3000)
clf.set_penalty(Penalty.l2(0.01))
clf.fit(X, y)

intercept, coefs = clf.params()
print(f"Fitted decision boundary: sigmoid({coefs[0]:.3f} * hours + {intercept:.3f})")

# --- Probabilities and accuracy ---------------------------------------------
probs = clf.predict(X)
accuracy = clf.score(y, probs)
print(f"Training accuracy: {accuracy:.3f}\n")

print("Probability of passing for a few study times:")
for h in [1.0, 3.0, 3.5, 4.0, 6.0]:
    p = clf.predict([[h]])[0]
    label = "PASS" if p >= 0.5 else "fail"
    print(f"  {h:4.1f} hours -> P(pass)={p:.3f}  ({label})")
