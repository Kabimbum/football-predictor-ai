from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/EnglandCSV.csv")

# ==========================
# PREPROCESS
# ==========================
def get_result(row):
    if row["FTH Goals"] > row["FTA Goals"]:
        return "Home Win"
    elif row["FTH Goals"] < row["FTA Goals"]:
        return "Away Win"
    else:
        return "Draw"

df["Result"] = df.apply(get_result, axis=1)

teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))

# ==========================
# FEATURE ENGINEERING
# ==========================
def avg_goals(team, side, col):
    return df[df[side] == team][col].mean()

def build_features(home, away):
    return pd.DataFrame([{
        "home_goals_avg": avg_goals(home, "HomeTeam", "FTH Goals"),
        "away_goals_avg": avg_goals(away, "AwayTeam", "FTA Goals"),
        "home_concede_avg": avg_goals(home, "HomeTeam", "FTA Goals"),
        "away_concede_avg": avg_goals(away, "AwayTeam", "FTH Goals"),
    }]).fillna(0)

X = []
y = []

for _, row in df.iterrows():
    X.append({
        "home_goals_avg": avg_goals(row["HomeTeam"], "HomeTeam", "FTH Goals"),
        "away_goals_avg": avg_goals(row["AwayTeam"], "AwayTeam", "FTA Goals"),
        "home_concede_avg": avg_goals(row["HomeTeam"], "HomeTeam", "FTA Goals"),
        "away_concede_avg": avg_goals(row["AwayTeam"], "AwayTeam", "FTH Goals"),
    })
    y.append(row["Result"])

X = pd.DataFrame(X).fillna(0)

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# MODELS
# ==========================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
knn = KNeighborsClassifier(n_neighbors=7)

rf.fit(X, y_enc)
xgb.fit(X, y_enc)
knn.fit(X_scaled, y_enc)

# ==========================
# FLASK
# ==========================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Football Predictor AI</title>
    <style>
        body { font-family: Arial; background:#111; color:#fff; padding:40px }
        select, button { padding:10px; margin:5px }
        .box { background:#222; padding:20px; margin-top:20px }
    </style>
</head>
<body>

<h2>Football Match Predictor</h2>

<select id="home">
    <option value="">Home Team</option>
    {% for t in teams %}
    <option value="{{t}}">{{t}}</option>
    {% endfor %}
</select>

<select id="away">
    <option value="">Away Team</option>
    {% for t in teams %}
    <option value="{{t}}">{{t}}</option>
    {% endfor %}
</select>

<button onclick="predict()">Predict</button>

<div class="box" id="result"></div>

<script>
function predict(){
    fetch("/predict", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body:JSON.stringify({
            home_team:document.getElementById("home").value,
            away_team:document.getElementById("away").value
        })
    })
    .then(res=>res.json())
    .then(d=>{
        if(d.error){
            document.getElementById("result").innerHTML = d.error;
            return;
        }
        document.getElementById("result").innerHTML = `
        <b>${d.home} vs ${d.away}</b><br><br>

        RF: ${d.RF_Result} (${d.RF_Confidence})<br>
        XGB: ${d.XGB_Result} (${d.XGB_Confidence})<br>
        KNN: ${d.KNN_Result} (${d.KNN_Confidence})<br><br>

        <b>Ensemble:</b> ${d.Ensemble_Result}<br>
        Confidence: ${d.Ensemble_Confidence}
        `;
    })
}
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, teams=teams)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home = data.get("home_team")
    away = data.get("away_team")

    if not home or not away:
        return jsonify({"error":"Pilih home dan away team"})

    if home == away:
        return jsonify({"error":"Home dan Away tidak boleh sama"})

    X_test = build_features(home, away)
    X_test_scaled = scaler.transform(X_test)

    rf_p = rf.predict(X_test)[0]
    xgb_p = xgb.predict(X_test)[0]
    knn_p = knn.predict(X_test_scaled)[0]

    rf_conf = float(np.max(rf.predict_proba(X_test)))
    xgb_conf = float(np.max(xgb.predict_proba(X_test)))
    knn_conf = float(np.max(knn.predict_proba(X_test_scaled)))

    votes = [
        le.inverse_transform([rf_p])[0],
        le.inverse_transform([xgb_p])[0],
        le.inverse_transform([knn_p])[0]
    ]

    ensemble = max(set(votes), key=votes.count)

    return jsonify({
        "home": home,
        "away": away,

        "RF_Result": votes[0],
        "RF_Confidence": round(rf_conf,3),

        "XGB_Result": votes[1],
        "XGB_Confidence": round(xgb_conf,3),

        "KNN_Result": votes[2],
        "KNN_Confidence": round(knn_conf,3),

        "Ensemble_Result": ensemble,
        "Ensemble_Confidence": round((rf_conf+xgb_conf+knn_conf)/3,3)
    })

if __name__ == "__main__":
    app.run(debug=True)
