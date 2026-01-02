import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ======================================================
# PATH CONFIG
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
DATA_PATH = os.path.join(BASE_DIR, "data", "EnglandCSVcleanded.csv")


# ======================================================
# LOAD DATA & MODELS
# ======================================================
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise Exception(f"[ERROR] Gagal load dataset: {e}")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    target_le = joblib.load(os.path.join(MODEL_DIR, "target_le.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    home_map = joblib.load(os.path.join(MODEL_DIR, "home_map.pkl"))
    away_map = joblib.load(os.path.join(MODEL_DIR, "away_map.pkl"))

    rf_clf = joblib.load(os.path.join(MODEL_DIR, "rf_clf.pkl"))
    xgb_clf = joblib.load(os.path.join(MODEL_DIR, "xgb_clf.pkl"))
    knn_clf = joblib.load(os.path.join(MODEL_DIR, "knn_clf.pkl"))

    rf_home = joblib.load(os.path.join(MODEL_DIR, "rf_goal_home.pkl"))
    rf_away = joblib.load(os.path.join(MODEL_DIR, "rf_goal_away.pkl"))
    xgb_home = joblib.load(os.path.join(MODEL_DIR, "xgb_goal_home.pkl"))
    xgb_away = joblib.load(os.path.join(MODEL_DIR, "xgb_goal_away.pkl"))
except Exception as e:
    raise Exception(f"[ERROR] Gagal load model: {e}")


# ======================================================
# GET AVERAGE STATISTICS
# ======================================================
def get_team_stats(home, away):

    home_df = df[df["HomeTeam"] == home]
    away_df = df[df["AwayTeam"] == away]

    stats = {
        "H Shots": home_df["H Shots"].mean(),
        "A Shots": away_df["A Shots"].mean(),
        "H SOT": home_df["H SOT"].mean(),
        "A SOT": away_df["A SOT"].mean(),
        "H Fouls": home_df["H Fouls"].mean(),
        "A Fouls": away_df["A Fouls"].mean(),
        "H Corners": home_df["H Corners"].mean(),
        "A Corners": away_df["A Corners"].mean(),
        "H Yellow": home_df["H Yellow"].mean(),
        "A Yellow": away_df["A Yellow"].mean(),
        "H Red": home_df["H Red"].mean(),
        "A Red": away_df["A Red"].mean(),
    }

    # isi median jika NaN
    for k in stats:
        if pd.isna(stats[k]):
            if k in df.columns:
                stats[k] = df[k].median()
            else:
                stats[k] = 0

    return stats


# ======================================================
# BUILD FEATURE VECTOR
# ======================================================
def build_features(home, away):

    if home not in home_map or away not in away_map:
        raise ValueError(f"Team tidak ditemukan: {home} / {away}")

    season_year = 2024
    stats = get_team_stats(home, away)

    numeric_order = [
        "H Shots","A Shots","H SOT","A SOT",
        "H Fouls","A Fouls","H Corners","A Corners",
        "H Yellow","A Yellow","H Red","A Red"
    ]

    row = []

    for col in feature_cols:
        if col == "Season_year":
            row.append(season_year)

        elif col in numeric_order:
            row.append(stats[col])

        elif col.startswith("home_prob"):
            idx = int(col.split("_")[-1])
            row.append(home_map[home][idx])

        elif col.startswith("away_prob"):
            idx = int(col.split("_")[-1])
            row.append(away_map[away][idx])

        else:
            row.append(0)

    return pd.DataFrame([row], columns=feature_cols)


# ======================================================
# FLASK APP
# ======================================================
app = Flask(__name__)



@app.route("/")
def index():
    teams = sorted(list(home_map.keys()))
    return render_template("index.html", teams=teams)


# ======================================================
# PREDICT ROUTE
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        home = request.form.get("home_team")
        away = request.form.get("away_team")

        if home == away:
            return jsonify({"error": "Home dan Away tidak boleh sama."})

        # Build features
        X = build_features(home, away)
        X_scaled = scaler.transform(X)

        # =========== CLASSIFICATION ===========
        rf_pred = target_le.inverse_transform(rf_clf.predict(X))[0]
        xgb_pred = target_le.inverse_transform(xgb_clf.predict(X))[0]
        knn_pred = target_le.inverse_transform(knn_clf.predict(X_scaled))[0]

        # =========== REGRESSION (GOALS) ===========
        home_rf = max(0, int(round(rf_home.predict(X)[0])))
        away_rf = max(0, int(round(rf_away.predict(X)[0])))

        home_xgb = max(0, int(round(xgb_home.predict(X)[0])))
        away_xgb = max(0, int(round(xgb_away.predict(X)[0])))

        # Ensemble score
        ensemble_home = max(0, int(round((home_rf + home_xgb) / 2)))
        ensemble_away = max(0, int(round((away_rf + away_xgb) / 2)))

        # ======================================================
        # FINAL WINNER (berdasarkan skor ensemble)
        # ======================================================
        if ensemble_home > ensemble_away:
            final_winner = "H"
        elif ensemble_home < ensemble_away:
            final_winner = "A"
        else:
            final_winner = "D"

        # -------------------------
        # Confidence (probability)
        # -------------------------
        try:
            rf_conf = float(np.max(rf_clf.predict_proba(X)))
        except Exception:
            rf_conf = None

        try:
            xgb_conf = float(np.max(xgb_clf.predict_proba(X)))
        except Exception:
            xgb_conf = None

        try:
            knn_conf = float(np.max(knn_clf.predict_proba(X_scaled)))
        except Exception:
            knn_conf = None

        # -------------------------
        # F1 proxy (safe single-sample proxy)
        # -------------------------
        try:
            rf_f1_proxy = float(rf_clf.score(X, target_le.transform([rf_pred])))
        except Exception:
            rf_f1_proxy = None

        try:
            xgb_f1_proxy = float(xgb_clf.score(X, target_le.transform([xgb_pred])))
        except Exception:
            xgb_f1_proxy = None

        try:
            knn_f1_proxy = float(knn_clf.score(X_scaled, target_le.transform([knn_pred])))
        except Exception:
            knn_f1_proxy = None

        # safe rounding helper
        def _r(v, nd=3):
            return round(v, nd) if (v is not None) else None

        return jsonify({
            "RandomForest": rf_pred,
            "XGBoost": xgb_pred,
            "KNN": knn_pred,

            "RF_score": f"{home_rf} - {away_rf}",
            "XGB_score": f"{home_xgb} - {away_xgb}",
            "Ensemble_score": f"{ensemble_home} - {ensemble_away}",

            "RF_conf": _r(rf_conf),
            "XGB_conf": _r(xgb_conf),
            "KNN_conf": _r(knn_conf),

            "RF_f1": _r(rf_f1_proxy),
            "XGB_f1": _r(xgb_f1_proxy),
            "KNN_f1": _r(knn_f1_proxy),

            # tambahan agar skor jelas
            "home_team": home,
            "away_team": away,
            "home_goals": ensemble_home,
            "away_goals": ensemble_away,

            "Final_Winner": final_winner
        })

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":

    app.run(debug=True)