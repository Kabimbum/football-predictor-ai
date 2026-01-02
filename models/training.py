import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance

# =====================================================
# PATH
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "EnglandCSV.csv")

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# =====================================================
# DATE PARSING (FIX WARNING)
# =====================================================
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
else:
    raise ValueError("Column 'Date' not found in dataset")

# =====================================================
# FILTER DATA 2013-2025
# =====================================================
df = df[(df["Date"].dt.year >= 2013) & (df["Date"].dt.year <= 2025)]
df = df.reset_index(drop=True)
print(f"Filtered shape (2013-2025): {df.shape}")

# =====================================================
# TARGET ENCODING
# =====================================================
le = LabelEncoder()
df["target"] = le.fit_transform(df["FT Result"])  # A, D, H

# =====================================================
# VISUALISASI DISTRIBUSI LABEL
# =====================================================
plt.figure()
df["FT Result"].value_counts().plot(kind="bar")
plt.title("Distribusi Hasil Pertandingan")
plt.xlabel("Result (H / D / A)")
plt.ylabel("Jumlah Match")
plt.show()

# =====================================================
# LONG FORMAT (HOME + AWAY)
# =====================================================
home = df[["Date", "HomeTeam", "FTH Goals", "FTA Goals", "FT Result"]].copy()
home.columns = ["Date", "Team", "GF", "GA", "Result"]
home["is_home"] = 1

away = df[["Date", "AwayTeam", "FTA Goals", "FTH Goals", "FT Result"]].copy()
away.columns = ["Date", "Team", "GF", "GA", "Result"]
away["is_home"] = 0

long_df = pd.concat([home, away]).sort_values("Date").reset_index(drop=True)

# =====================================================
# POINTS CALCULATION
# =====================================================
def calc_points(row):
    if row["Result"] == "D":
        return 1
    if row["is_home"] == 1 and row["Result"] == "H":
        return 3
    if row["is_home"] == 0 and row["Result"] == "A":
        return 3
    return 0

long_df["Points"] = long_df.apply(calc_points, axis=1)

# =====================================================
# ROLLING FEATURES (LAST 5 MATCHES)
# =====================================================
long_df = long_df.sort_values(["Team", "Date"])

long_df["form"] = (
    long_df.groupby("Team")["Points"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

long_df["gf_avg"] = (
    long_df.groupby("Team")["GF"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

long_df["ga_avg"] = (
    long_df.groupby("Team")["GA"]
    .transform(lambda x: x.rolling(5, min_periods=1).mean())
)

# =====================================================
# MERGE BACK TO MATCH DATA
# =====================================================
home_feat = long_df[long_df["is_home"] == 1][
    ["Date", "Team", "form", "gf_avg", "ga_avg"]
]

away_feat = long_df[long_df["is_home"] == 0][
    ["Date", "Team", "form", "gf_avg", "ga_avg"]
]

df = df.merge(
    home_feat,
    left_on=["Date", "HomeTeam"],
    right_on=["Date", "Team"],
    how="left"
)

df = df.merge(
    away_feat,
    left_on=["Date", "AwayTeam"],
    right_on=["Date", "Team"],
    how="left",
    suffixes=("_home", "_away")
)

# =====================================================
# FEATURES
# =====================================================
FEATURES = [
    "form_home", "form_away",
    "gf_avg_home", "gf_avg_away",
    "ga_avg_home", "ga_avg_away"
]

df_model = df.dropna(subset=FEATURES + ["target"])

X = df_model[FEATURES]
y = df_model["target"]
y_home_goals = df_model["FTH Goals"]
y_away_goals = df_model["FTA Goals"]

# =====================================================
# TIME-BASED SPLIT
# =====================================================
split_idx = int(len(df_model) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

y_home_train = y_home_goals.iloc[:split_idx]
y_home_test = y_home_goals.iloc[split_idx:]

y_away_train = y_away_goals.iloc[:split_idx]
y_away_test = y_away_goals.iloc[split_idx:]

# =====================================================
# SCALING (FOR KNN)
# =====================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =====================================================
# CLASSIFICATION MODELS
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train_s, y_train)
knn_pred = knn.predict(X_test_s)

# =====================================================
# REGRESSION MODELS FOR SCORE PREDICTION
# =====================================================
print("\n=== TRAINING SCORE PREDICTION MODELS ===")

# Model untuk prediksi gol Home
xgb_home_score = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_home_score.fit(X_train, y_home_train)
home_pred = xgb_home_score.predict(X_test)

# Model untuk prediksi gol Away
xgb_away_score = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_away_score.fit(X_train, y_away_train)
away_pred = xgb_away_score.predict(X_test)

# Evaluasi Score Prediction
home_mae = mean_absolute_error(y_home_test, home_pred)
away_mae = mean_absolute_error(y_away_test, away_pred)

# Round predictions untuk exact match accuracy
home_pred_rounded = np.round(home_pred)
away_pred_rounded = np.round(away_pred)

# Exact score accuracy
home_exact_acc = accuracy_score(y_home_test, home_pred_rounded)
away_exact_acc = accuracy_score(y_away_test, away_pred_rounded)

# Combined exact score accuracy (kedua skor harus tepat)
exact_match = ((home_pred_rounded == y_home_test.values) & 
               (away_pred_rounded == y_away_test.values))
exact_score_acc = exact_match.sum() / len(y_home_test)

# Goal difference accuracy
pred_diff = home_pred_rounded - away_pred_rounded
actual_diff = y_home_test.values - y_away_test.values
diff_acc = accuracy_score(np.sign(actual_diff), np.sign(pred_diff))

# Within 1 goal accuracy
home_within_1 = np.abs(home_pred_rounded - y_home_test.values) <= 1
away_within_1 = np.abs(away_pred_rounded - y_away_test.values) <= 1
within_1_acc = (home_within_1 & away_within_1).sum() / len(y_home_test)

print("\n" + "="*60)
print("SCORE PREDICTION PERFORMANCE REPORT")
print("="*60)
print(f"\n📊 Mean Absolute Error (MAE):")
print(f"   Home Goals MAE: {home_mae:.3f}")
print(f"   Away Goals MAE: {away_mae:.3f}")
print(f"   Average MAE   : {(home_mae + away_mae)/2:.3f}")

print(f"\n🎯 Exact Goal Accuracy:")
print(f"   Home Goals Exact: {home_exact_acc*100:.2f}%")
print(f"   Away Goals Exact: {away_exact_acc*100:.2f}%")

print(f"\n⚽ Exact Score Match:")
print(f"   Both scores correct: {exact_score_acc*100:.2f}%")

print(f"\n📈 Goal Difference Accuracy:")
print(f"   Correct winner/draw: {diff_acc*100:.2f}%")

print(f"\n✅ Within 1 Goal Accuracy:")
print(f"   Both within ±1 goal: {within_1_acc*100:.2f}%")

# Distribution of prediction errors
print(f"\n📉 Error Distribution:")
home_errors = np.abs(home_pred_rounded - y_home_test.values)
away_errors = np.abs(away_pred_rounded - y_away_test.values)
print(f"   Home - 0 error: {(home_errors == 0).sum()/len(home_errors)*100:.1f}%")
print(f"   Home - 1 error: {(home_errors == 1).sum()/len(home_errors)*100:.1f}%")
print(f"   Home - 2+ error: {(home_errors >= 2).sum()/len(home_errors)*100:.1f}%")
print(f"   Away - 0 error: {(away_errors == 0).sum()/len(away_errors)*100:.1f}%")
print(f"   Away - 1 error: {(away_errors == 1).sum()/len(away_errors)*100:.1f}%")
print(f"   Away - 2+ error: {(away_errors >= 2).sum()/len(away_errors)*100:.1f}%")
print("="*60)

# =====================================================
# ACCURACY
# =====================================================
print("\n=== CLASSIFICATION ACCURACY ===")
print("RF :", accuracy_score(y_test, rf_pred))
print("XGB:", accuracy_score(y_test, xgb_pred))
print("KNN:", accuracy_score(y_test, knn_pred))

# =====================================================
# CLASSIFICATION REPORTS
# =====================================================
print("\n=== CLASSIFICATION REPORT (Random Forest) ===")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

print("\n=== CLASSIFICATION REPORT (XGBoost) ===")
print(classification_report(y_test, xgb_pred, target_names=le.classes_))

print("\n=== CLASSIFICATION REPORT (KNN) ===")
print(classification_report(y_test, knn_pred, target_names=le.classes_))

# =====================================================
# CONFIDENCE (AVERAGE MAX PROBABILITY)
# =====================================================
rf_conf = np.mean(np.max(rf.predict_proba(X_test), axis=1))
xgb_conf = np.mean(np.max(xgb.predict_proba(X_test), axis=1))
knn_conf = np.mean(np.max(knn.predict_proba(X_test_s), axis=1))

print("\n=== MODEL CONFIDENCE ===")
print(f"Random Forest : {rf_conf:.3f}")
print(f"XGBoost       : {xgb_conf:.3f}")
print(f"KNN           : {knn_conf:.3f}")

# =====================================================
# VISUALISASI AKURASI
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification Accuracy
axes[0].bar(
    ["Random Forest", "XGBoost", "KNN"],
    [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred),
        accuracy_score(y_test, knn_pred)
    ]
)
axes[0].set_title("Perbandingan Akurasi Model Klasifikasi")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")

# Score Prediction Accuracy
axes[1].bar(
    ["Score Sesuai", "Selisih 1", "Lebih ±1"],
    [exact_score_acc, diff_acc, within_1_acc],
    color=['#ff6b6b', '#4ecdc4', '#45b7d1']
)
axes[1].set_title("Perbandingan Akurasi Prediksi Skor")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

# =====================================================
# CONFUSION MATRIX (XGB)
# =====================================================
cm = confusion_matrix(y_test, xgb_pred)

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xticks(range(3), le.classes_)
plt.yticks(range(3), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Prediksi Random Forest
rf_pred = rf.predict(X_test)

# =====================================================
# CONFUSION MATRIX (Random forest)
# =====================================================

# Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm_rf)
plt.colorbar()
plt.xticks(range(3), le.classes_)
plt.yticks(range(3), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm_rf[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# =====================================================
# CONFUSION MATRIX (KNN)
# =====================================================

# Prediksi KNN (pakai data yang sudah di-scale)
knn_pred = knn.predict(X_test_s)

# Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm_knn)
plt.colorbar()
plt.xticks(range(3), le.classes_)
plt.yticks(range(3), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm_knn[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()


# =====================================================
# FEATURE IMPORTANCE (XGBOOST)
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Classification Feature Importance
axes[0].barh(FEATURES, xgb.feature_importances_)
axes[0].set_title("Feature Importance - Classification (XGBoost)")
axes[0].set_xlabel("Importance")

# Home Score Feature Importance
axes[1].barh(FEATURES, xgb_home_score.feature_importances_)
axes[1].set_title("Feature Importance - Home Goals (XGBoost)")
axes[1].set_xlabel("Importance")

# Away Score Feature Importance
axes[2].barh(FEATURES, xgb_away_score.feature_importances_)
axes[2].set_title("Feature Importance - Away Goals (XGBoost)")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.show()

# =====================================================
# FEATURE IMPORTANCE (random forest)
# =====================================================

importances = rf.feature_importances_

plt.figure(figsize=(8, 4))
plt.barh(FEATURES, importances)
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# =====================================================
# FEATURE IMPORTANCE (KNN)
# =====================================================

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    knn,
    X_test_s,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

knn_importance = perm.importances_mean

plt.figure(figsize=(8, 4))
plt.barh(FEATURES, knn_importance)
plt.xlabel("Decrease in Accuracy")
plt.title("Feature Importance - KNN (Permutation)")
plt.tight_layout()
plt.show()


# =====================================================
# VISUALISASI SCORE PREDICTION ERROR
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Home Goals: Actual vs Predicted
axes[0].scatter(y_home_test, home_pred_rounded, alpha=0.5)
axes[0].plot([0, y_home_test.max()], [0, y_home_test.max()], 'r--', lw=2)
axes[0].set_xlabel("Actual Home Goals")
axes[0].set_ylabel("Predicted Home Goals")
axes[0].set_title("Home Goals: Actual vs Predicted")
axes[0].grid(True, alpha=0.3)

# Away Goals: Actual vs Predicted
axes[1].scatter(y_away_test, away_pred_rounded, alpha=0.5, color='orange')
axes[1].plot([0, y_away_test.max()], [0, y_away_test.max()], 'r--', lw=2)
axes[1].set_xlabel("Actual Away Goals")
axes[1].set_ylabel("Predicted Away Goals")
axes[1].set_title("Away Goals: Actual vs Predicted")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# FUNCTION FOR PREDICTION
# =====================================================
def get_team_stats(team_name, long_df):
    """Ambil statistik terbaru dari tim"""
    team_data = long_df[long_df["Team"] == team_name].tail(5)
    
    if len(team_data) == 0:
        return None
    
    stats = {
        "form": team_data["Points"].mean(),
        "gf_avg": team_data["GF"].mean(),
        "ga_avg": team_data["GA"].mean()
    }
    
    return stats

def predict_match(home_team, away_team, long_df, xgb_classifier, 
                  xgb_home_reg, xgb_away_reg, scaler, le):
    """Prediksi hasil pertandingan dan skor"""
    
    # Ambil statistik tim
    home_stats = get_team_stats(home_team, long_df)
    away_stats = get_team_stats(away_team, long_df)
    
    if home_stats is None:
        print(f"Error: Tim '{home_team}' tidak ditemukan dalam dataset")
        return None
    
    if away_stats is None:
        print(f"Error: Tim '{away_team}' tidak ditemukan dalam dataset")
        return None
    
    # Buat feature vector
    features = np.array([[
        home_stats["form"],
        away_stats["form"],
        home_stats["gf_avg"],
        away_stats["gf_avg"],
        home_stats["ga_avg"],
        away_stats["ga_avg"]
    ]])
    
    # Prediksi hasil (H/D/A)
    result_pred = xgb_classifier.predict(features)[0]
    result_proba = xgb_classifier.predict_proba(features)[0]
    result_label = le.inverse_transform([result_pred])[0]
    
    # Prediksi skor
    home_score = max(0, round(xgb_home_reg.predict(features)[0]))
    away_score = max(0, round(xgb_away_reg.predict(features)[0]))
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "result": result_label,
        "result_proba": {
            "H": result_proba[le.transform(["H"])[0]],
            "D": result_proba[le.transform(["D"])[0]],
            "A": result_proba[le.transform(["A"])[0]]
        },
        "score": f"{home_score}-{away_score}",
        "home_score": home_score,
        "away_score": away_score
    }

# =====================================================
# CLI INTERFACE
# =====================================================
def main_cli():
    """Command Line Interface untuk prediksi pertandingan"""
    
    # Ambil daftar tim unik
    teams = sorted(long_df["Team"].unique())
    
    print("\n" + "="*60)
    print("FOOTBALL MATCH PREDICTION SYSTEM")
    print("="*60)
    
    while True:
        print("\n--- DAFTAR TIM TERSEDIA ---")
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team}")
        
        print("\n" + "-"*60)
        
        # Input Home Team
        while True:
            home_input = input("\nMasukkan nama Home Team (atau nomor, atau 'q' untuk keluar): ").strip()
            
            if home_input.lower() == 'q':
                print("Terima kasih! Program selesai.")
                return
            
            # Cek apakah input adalah angka
            if home_input.isdigit():
                idx = int(home_input) - 1
                if 0 <= idx < len(teams):
                    home_team = teams[idx]
                    break
                else:
                    print(f"Error: Nomor harus antara 1-{len(teams)}")
            else:
                # Cari tim yang cocok (case insensitive)
                matches = [t for t in teams if home_input.lower() in t.lower()]
                if len(matches) == 1:
                    home_team = matches[0]
                    break
                elif len(matches) > 1:
                    print(f"Beberapa tim ditemukan: {', '.join(matches)}")
                    print("Silakan spesifikan lebih detail.")
                else:
                    print("Tim tidak ditemukan. Silakan coba lagi.")
        
        # Input Away Team
        while True:
            away_input = input("\nMasukkan nama Away Team (atau nomor, atau 'q' untuk keluar): ").strip()
            
            if away_input.lower() == 'q':
                print("Terima kasih! Program selesai.")
                return
            
            # Cek apakah input adalah angka
            if away_input.isdigit():
                idx = int(away_input) - 1
                if 0 <= idx < len(teams):
                    away_team = teams[idx]
                    if away_team == home_team:
                        print("Error: Away team tidak boleh sama dengan Home team")
                        continue
                    break
                else:
                    print(f"Error: Nomor harus antara 1-{len(teams)}")
            else:
                # Cari tim yang cocok (case insensitive)
                matches = [t for t in teams if away_input.lower() in t.lower()]
                if len(matches) == 1:
                    away_team = matches[0]
                    if away_team == home_team:
                        print("Error: Away team tidak boleh sama dengan Home team")
                        continue
                    break
                elif len(matches) > 1:
                    print(f"Beberapa tim ditemukan: {', '.join(matches)}")
                    print("Silakan spesifikan lebih detail.")
                else:
                    print("Tim tidak ditemukan. Silakan coba lagi.")
        
        # Prediksi
        print("\n" + "="*60)
        print("MELAKUKAN PREDIKSI...")
        print("="*60)
        
        result = predict_match(
            home_team, 
            away_team, 
            long_df, 
            xgb, 
            xgb_home_score, 
            xgb_away_score, 
            scaler, 
            le
        )
        
        if result:
            print(f"\n🏟️  PERTANDINGAN: {result['home_team']} vs {result['away_team']}")
            print("-"*60)
            print(f"\n⚽ PREDIKSI SKOR: {result['score']}")
            print(f"   ({result['home_team']} {result['home_score']} - {result['away_score']} {result['away_team']})")
            
            print(f"\n🏆 PREDIKSI PEMENANG:")
            if result['result'] == 'H':
                print(f"   {result['home_team']} (Home Win)")
            elif result['result'] == 'A':
                print(f"   {result['away_team']} (Away Win)")
            else:
                print("   Draw (Seri)")
            
            print(f"\n📊 PROBABILITAS:")
            print(f"   Home Win: {result['result_proba']['H']*100:.1f}%")
            print(f"   Draw    : {result['result_proba']['D']*100:.1f}%")
            print(f"   Away Win: {result['result_proba']['A']*100:.1f}%")
        
        # Tanya apakah ingin prediksi lagi
        print("\n" + "="*60)
        lanjut = input("\nPrediksi pertandingan lain? (y/n): ").strip().lower()
        if lanjut != 'y':
            print("\nTerima kasih! Program selesai.")
            break

# =====================================================
# JALANKAN CLI
# =====================================================
if __name__ == "__main__":
    main_cli()