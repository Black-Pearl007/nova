"""Train and evaluate Nova's intent classifier, compare it with the rule-based baseline,
and export the model to ../model.js so the browser version can use it.

Run:  python train.py
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

import baseline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42


def prep(s: str) -> str:
    """Lowercase and drop apostrophes so "i'm" becomes "im" (same logic is used in the browser)."""
    return s.lower().replace("’", "'").replace("'", "")


def read(name, cols):
    with open(os.path.join(DATA, name), encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    return [[r[c] for r in rows] for c in cols]


X_train, y_train = read("train.csv", ["text", "label"])
X_test, y_test = read("test.csv", ["text", "label"])
(X_oos,) = read("out_of_scope.csv", ["text"])
labels = sorted(set(y_train))
print(f"train={len(X_train)} test={len(X_test)} out-of-scope={len(X_oos)} intents={len(labels)}")


def word_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=10, max_iter=2000, random_state=SEED)),
    ])


def word_char_model():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5))),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=10, max_iter=2000, random_state=SEED))])


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
    }


results = {}

# 1. Rule-based baseline on the held-out test set
base_pred = [baseline.predict(t) for t in X_test]
results["baseline_rules"] = scores(y_test, base_pred)
results["baseline_rules"]["unmatched"] = sum(p == "unknown" for p in base_pred)

# 2. Candidate models: 5-fold cross-validation on TRAIN only, then one evaluation on the held-out test set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {"tfidf_word_lr": word_model, "tfidf_word_char_lr": word_char_model}
fitted, test_pred = {}, {}
for name, make in models.items():
    cv_pred = cross_val_predict(make(), X_train, y_train, cv=cv)
    m = make().fit(X_train, y_train)
    fitted[name] = m
    test_pred[name] = list(m.predict(X_test))
    results[name] = {"cv_on_train": scores(y_train, cv_pred), "test": scores(y_test, test_pred[name])}

# Model selection uses cross-validation on TRAIN only, never the test set
EXPORT = max(models, key=lambda n: results[n]["cv_on_train"]["macro_f1"])
model = fitted[EXPORT]
pred = test_pred[EXPORT]
print("selected by cross-validation:", EXPORT)

# Learning curve: how accuracy grows with the amount of training data per intent
rng = np.random.RandomState(SEED)
by_label = {l: [i for i, y in enumerate(y_train) if y == l] for l in labels}
curve_sizes = [4, 8, 12, 16]
lc = {n: [] for n in models}
for k in curve_sizes:
    accs = {n: [] for n in models}
    for rep in range(5):
        idx = [i for l in labels for i in rng.choice(by_label[l], k, replace=False)]
        xs, ys = [X_train[i] for i in idx], [y_train[i] for i in idx]
        for n, make in models.items():
            accs[n].append(accuracy_score(y_test, make().fit(xs, ys).predict(X_test)))
    for n in models:
        lc[n].append(round(float(np.mean(accs[n])), 3))
results["learning_curve"] = {"examples_per_intent": curve_sizes, **lc, "baseline_rules": results["baseline_rules"]["accuracy"]}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(curve_sizes, lc["tfidf_word_lr"], "o-", label="TF-IDF words + LR")
ax.plot(curve_sizes, lc["tfidf_word_char_lr"], "s-", label="TF-IDF words+chars + LR")
ax.axhline(results["baseline_rules"]["accuracy"], color="#999", ls="--", label="Rule-based baseline")
ax.set_xlabel("Training examples per intent")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0, 1)
ax.set_title("More data keeps helping (mean of 5 random subsets)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "learning_curve.png"), dpi=140)
plt.close(fig)

# 3. Confidence threshold: accept in-domain test messages, reject out-of-scope ones
proba_test = model.predict_proba(X_test).max(axis=1)
proba_oos = model.predict_proba(X_oos).max(axis=1)
curve = []
for t in np.arange(0.10, 0.71, 0.05):
    kept = proba_test >= t
    acc_when_kept = float(np.mean([p == y for p, y, k in zip(pred, y_test, kept) if k])) if kept.any() else 0.0
    curve.append({
        "threshold": round(float(t), 2),
        "in_scope_answered": round(float(kept.mean()), 3),
        "accuracy_when_answered": round(acc_when_kept, 3),
        "out_of_scope_rejected": round(float((proba_oos < t).mean()), 3),
    })
# choose the threshold with the best balance of answering real intents and rejecting off-topic text
best = max(curve, key=lambda c: c["in_scope_answered"] * c["out_of_scope_rejected"])
THRESHOLD = best["threshold"]
results["threshold"] = best
results["threshold_curve"] = curve

# 4. Per-class report, errors, plots
report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
base_report = classification_report(y_test, base_pred, labels=labels, output_dict=True, zero_division=0)
results["per_class_f1"] = {l: {"baseline": round(base_report[l]["f1-score"], 3), "model": round(report[l]["f1-score"], 3)} for l in labels}

with open(os.path.join(OUT, "errors.csv"), "w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["text", "true", "model_pred", "model_confidence", "baseline_pred"])
    for t, y, p, c, b in zip(X_test, y_test, pred, proba_test, base_pred):
        if p != y or b != y:
            w.writerow([t, y, p, round(float(c), 2), b])

cm = confusion_matrix(y_test, pred, labels=labels)
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Intent classifier: confusion matrix on held-out test set")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(labels))
ax.bar(x - 0.2, [results["per_class_f1"][l]["baseline"] for l in labels], 0.4, label="Rule-based baseline", color="#bbbbbb")
ax.bar(x + 0.2, [results["per_class_f1"][l]["model"] for l in labels], 0.4, label="TF-IDF + logistic regression", color="#2a9df4")
ax.set_xticks(x, labels, rotation=60, ha="right")
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1.05)
ax.set_title("Per-intent F1 on held-out test set")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_intent_f1.png"), dpi=140)
plt.close(fig)

# 5. Export to JavaScript
tf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
vecs = [(n, v) for n, v in tf.transformer_list] if isinstance(tf, FeatureUnion) else [("word", tf)]
blocks = []
for n, v in vecs:
    blocks.append({
        "analyzer": v.analyzer if isinstance(v.analyzer, str) else "word",
        "ngram": list(v.ngram_range),
        "terms": v.get_feature_names_out().tolist(),
        "idf": [round(float(x), 5) for x in v.idf_],
    })
terms = [t for b in blocks for t in b["terms"]]
payload = {
    "labels": clf.classes_.tolist(),
    "blocks": blocks,
    "coef": [[round(float(v), 3) for v in row] for row in clf.coef_],
    "intercept": [round(float(v), 4) for v in clf.intercept_],
    "threshold": THRESHOLD,
    "note": "TF-IDF (each block l2-normalised, concatenated) + multinomial logistic regression. Trained by nlp/train.py",
}
js_path = os.path.join(HERE, "..", "model.js")
with open(js_path, "w", encoding="utf8") as f:
    f.write("window.NOVA_MODEL=" + json.dumps(payload, separators=(",", ":")) + ";\n")
# parity fixture so the JavaScript implementation can be checked against scikit-learn
fixture = X_test + X_oos
fx = model.predict_proba(fixture)
with open(os.path.join(OUT, "parity_fixture.json"), "w", encoding="utf8") as f:
    json.dump([{"text": t, "proba": [round(float(v), 5) for v in p]} for t, p in zip(fixture, fx)], f)

results["dataset"] = {"train": len(X_train), "test": len(X_test), "out_of_scope": len(X_oos), "intents": len(labels), "vocab_size": len(terms)}
results["exported_model"] = EXPORT
with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf8") as f:
    json.dump(results, f, indent=2)

print(json.dumps({k: results[k] for k in ["baseline_rules", "tfidf_word_lr", "tfidf_word_char_lr", "threshold", "dataset"]}, indent=2))
print("model.js size KB:", round(os.path.getsize(js_path) / 1024, 1))
print("\nTest errors of the exported model:")
for t, y, p, c in zip(X_test, y_test, pred, proba_test):
    if p != y:
        print(f"  '{t}' true={y} predicted={p} ({c:.2f})")
