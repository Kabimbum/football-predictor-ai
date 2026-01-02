import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no GUI (WSL-safe)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = "/root/college/AI/footballpred/football-predictor-ai/data/EnglandCSVcleanded.csv"
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models/saved_models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading dataset:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)

# ---------------- LOAD ARTIFACTS ----------------
print("Loading saved artifacts from:", MODEL_DIR)
needed = ["feature_cols.pkl", "home_map.pkl", "away_map.pkl", "scaler.pkl", "target_le.pkl"]
for f in needed:
    path = os.path.join(MODEL_DIR, f)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact missing: {path}")

feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
home_map = joblib.load(os.path.join(MODEL_DIR, "home_map.pkl"))
away_map = joblib.load(os.path.join(MODEL_DIR, "away_map.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
target_le = joblib.load(os.path.join(MODEL_DIR, "target_le.pkl"))

# load classifiers (optional: fail gracefully)
def try_load(name):
    p = os.path.join(MODEL_DIR, name)
    return joblib.load(p) if os.path.exists(p) else None

rf_clf = try_load("rf_clf.pkl")
xgb_clf = try_load("xgb_clf.pkl")
knn_clf = try_load("knn_clf.pkl")

rf_goal_home = try_load("rf_goal_home.pkl") or try_load("rf_goal_home.pkl") or None
rf_goal_away = try_load("rf_goal_away.pkl") or None
xgb_goal_home = try_load("xgb_goal_home.pkl")
xgb_goal_away = try_load("xgb_goal_away.pkl")

print("Artifacts loaded. Models present:",
      f"RF clf={'yes' if rf_clf is not None else 'no'}",
      f"XGB clf={'yes' if xgb_clf is not None else 'no'}",
      f"KNN={'yes' if knn_clf is not None else 'no'}",
      f"RF_reg_home={'yes' if rf_goal_home is not None else 'no'}",
      f"XGB_reg_home={'yes' if xgb_goal_home is not None else 'no'}")

# ---------------- REBUILD FEATURES (same as training.py) ----------------
def season_to_year(season_str):
    a, b = str(season_str).split("/")
    return 2000 + int(b)

df["Season_year"] = df["Season"].apply(season_to_year)
df = df[df["Season_year"].between(2020, 2025)]

# target encoding using saved encoder
df["target"] = target_le.transform(df["FT Result"])

# identify home/away prob columns from feature_cols saved
home_cols = [c for c in feature_cols if c.startswith("home_prob")]
away_cols = [c for c in feature_cols if c.startswith("away_prob")]

# build home/away prob arrays robustly (use zeros if team missing)
def get_prob(map_obj, team, n):
    return map_obj.get(team, np.zeros(n))

n_probs = len(home_cols)  # should equal len(target_le.classes_)
home_prob_arr = []
away_prob_arr = []
for _, row in df.iterrows():
    home_prob_arr.append(get_prob(home_map, row["HomeTeam"], n_probs))
    away_prob_arr.append(get_prob(away_map, row["AwayTeam"], n_probs))

# assign arrays to dataframe (will expand across columns)
df[home_cols] = np.array(home_prob_arr)
df[away_cols] = np.array(away_prob_arr)

# numeric columns that training used:
numeric_cols = [
    'H Shots','A Shots','H SOT','A SOT',
    'H Fouls','A Fouls','H Corners','A Corners',
    'H Yellow','A Yellow','H Red','A Red'
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

# final X and scaled X for KNN
X = df[feature_cols]
X_scaled = scaler.transform(X)
y_true = df["target"]

# -------------- HELPERS: SAVE PLOTS --------------
def save_confusion_matrix(model, name, X_in):
    if model is None:
        print(f"Skipping {name} — model not found.")
        return
    print("Saving CM:", name)
    preds = model.predict(X_in)
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_le.classes_)
    disp.plot(cmap="Blues")
    plt.title(name)
    out = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", out)

def save_feature_importance(model, name):
    if model is None or not hasattr(model, "feature_importances_"):
        print(f"Skipping FI: {name} (no model or no feature_importances_)")
        return
    print("Saving Feature Importance:", name)
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values().tail(20)
    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.title(name)
    out = os.path.join(PLOT_DIR, f"{name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", out)

def save_roc_curve(model, name, X_in):
    if model is None:
        print(f"Skipping ROC: {name} (model missing)")
        return
    if not hasattr(model, "predict_proba"):
        print(f"Skipping ROC: {name} (predict_proba not available)")
        return
    print("Saving ROC:", name)
    # binarize encoded labels (0,1,2)
    y_bin = label_binarize(y_true, classes=list(range(len(target_le.classes_))))
    y_score = model.predict_proba(X_in)
    plt.figure(figsize=(8,6))
    for i, cl in enumerate(target_le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cl} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {name}")
    plt.legend(loc="lower right")
    out = os.path.join(PLOT_DIR, f"ROC_{name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", out)

def save_regression_plots(reg_model, true_series, X_in, prefix):
    if reg_model is None:
        print(f"Skipping regression plots: {prefix} (model missing)")
        return
    # true vs pred scatter
    print("Saving regression scatter:", prefix)
    pred = reg_model.predict(X_in)
    plt.figure(figsize=(6,6))
    plt.scatter(true_series, pred, alpha=0.6)
    plt.plot([0, max(true_series)], [0, max(true_series)], "r--")
    plt.xlabel("True Goals")
    plt.ylabel("Predicted Goals")
    plt.title(f"{prefix} True vs Pred")
    out1 = os.path.join(PLOT_DIR, f"TvsP_{prefix}.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", out1)

    # error distribution
    print("Saving regression error distribution:", prefix)
    errors = pred - true_series
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=30)
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")
    plt.title(f"{prefix} Error Distribution")
    out2 = os.path.join(PLOT_DIR, f"ErrDist_{prefix}.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", out2)

# ---------------- RUN & SAVE ----------------
print("\n--- Saving classification confusion matrices ---")
save_confusion_matrix(rf_clf, "CM_RandomForest", X)
save_confusion_matrix(xgb_clf, "CM_XGBoost", X)
# KNN needs scaled X
save_confusion_matrix(knn_clf, "CM_KNN", X_scaled)

print("\n--- Saving feature importances ---")
save_feature_importance(rf_clf, "FI_RandomForest")
save_feature_importance(xgb_clf, "FI_XGBoost")

print("\n--- Saving ROC curves (classifiers) ---")
save_roc_curve(rf_clf, "RandomForest", X)
save_roc_curve(xgb_clf, "XGBoost", X)
save_roc_curve(knn_clf, "KNN", X_scaled)

print("\n--- Saving regression plots ---")
# prefer RF regressors if present, otherwise XGB regressors (we attempt both)
save_regression_plots(rf_goal_home, df["FTH Goals"], X, "RF_HomeGoals")
save_regression_plots(rf_goal_away, df["FTA Goals"], X, "RF_AwayGoals")
save_regression_plots(xgb_goal_home, df["FTH Goals"], X, "XGB_HomeGoals")
save_regression_plots(xgb_goal_away, df["FTA Goals"], X, "XGB_AwayGoals")

print("\nAll plots saved to:", PLOT_DIR)
print("Done.")
