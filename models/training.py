import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import permutation_importance

# =====================================================
# PATH SETUP
# =====================================================

# Mendapatkan direktori root proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Menentukan path file CSV di folder 'data' di dalam root proyek
DATA_PATH = os.path.join(BASE_DIR, "data", "EnglandCSV.csv")




# Menampilkan path file yang akan dibaca (untuk memastikan path benar)
print("Loading dataset from:", DATA_PATH)

# Membaca file CSV ke dalam DataFrame pandas
df = pd.read_csv(DATA_PATH)

# Menampilkan jumlah baris dan kolom dataset awal
print("Raw shape:", df.shape)

# =====================================================
# DATE PARSING (HANDLE DATE FORMAT & PREVENT WARNINGS)
# =====================================================

# Mengecek apakah kolom "Date" ada di dalam DataFrame
if "Date" in df.columns:

    # Mengonversi kolom "Date" menjadi tipe datetime
    # - dayfirst=True  → format tanggal diasumsikan DD-MM-YYYY
    # - errors="coerce" → nilai tanggal tidak valid akan diubah menjadi NaT
    #   (Not a Time) agar tidak menyebabkan crash pada program
    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Menghapus baris yang memiliki nilai Date = NaT
    # Baris ini biasanya berasal dari data rusak atau format tanggal tidak konsisten
    df = df.dropna(subset=["Date"])

    # Mengurutkan data berdasarkan tanggal secara kronologis (ascending)
    # Hal ini penting untuk analisis time-series atau data berbasis urutan waktu
    df = df.sort_values("Date")

    # Mereset index DataFrame setelah penghapusan dan pengurutan data
    # drop=True → index lama tidak disimpan sebagai kolom baru
    df = df.reset_index(drop=True)

else:
    # Jika kolom "Date" tidak ditemukan,
    # program dihentikan dan menampilkan pesan error yang jelas
    raise ValueError("Column 'Date' not found in dataset")


# =====================================================
# FILTER DATA BERDASARKAN RENTANG TAHUN (2013–2025)
# =====================================================

# Melakukan penyaringan data berdasarkan kolom "Date"
# Hanya baris dengan tahun antara 2013 sampai 2025 yang dipertahankan
# - df["Date"].dt.year digunakan untuk mengekstrak informasi tahun dari tipe datetime
# - Operator & digunakan untuk menggabungkan dua kondisi logika
df = df[
    (df["Date"].dt.year >= 2013) &
    (df["Date"].dt.year <= 2025)
]

# Mereset index DataFrame setelah proses filtering
# drop=True memastikan index lama tidak disimpan sebagai kolom baru
df = df.reset_index(drop=True)

# Menampilkan ukuran DataFrame hasil filtering
# Informasi ini berguna untuk validasi bahwa proses filtering berhasil
print(f"Filtered shape (2013–2025): {df.shape}")

# =====================================================
# TARGET ENCODING (LABEL ENCODER)
# =====================================================

# Membuat objek LabelEncoder dari sklearn
# LabelEncoder digunakan untuk mengubah label kategorikal
# (dalam hal ini hasil pertandingan: A, D, H)
# menjadi representasi numerik yang dapat diproses oleh model machine learning
le = LabelEncoder()

# Mengonversi kolom "FT Result" menjadi label numerik
# Contoh hasil encoding (tergantung urutan alfabet):
# A → 0, D → 1, H → 2
# Kolom baru bernama "target" akan digunakan sebagai variabel target (y)
df["target"] = le.fit_transform(df["FT Result"])  # A, D, H

# =====================================================
# VISUALISASI DISTRIBUSI LABEL (CLASS DISTRIBUTION)
# =====================================================

# Membuat figure baru untuk plot
# plt.figure() dipanggil tanpa ukuran khusus agar default Matplotlib digunakan
plt.figure()

# Menghitung jumlah kemunculan masing-masing kelas
# kemudian divisualisasikan dalam bentuk diagram batang (bar chart)
df["FT Result"].value_counts().plot(kind="bar")

# Memberikan judul pada grafik
# Judul ini menjelaskan bahwa grafik menunjukkan distribusi hasil pertandingan
plt.title("Distribusi Hasil Pertandingan")

# Memberikan label pada sumbu X
# Menunjukkan kategori hasil pertandingan:
# H = Home Win, D = Draw, A = Away Win
plt.xlabel("Result (H / D / A)")

# Memberikan label pada sumbu Y
# Menunjukkan jumlah pertandingan pada masing-masing kelas
plt.ylabel("Jumlah Match")

# Menampilkan grafik ke layar
plt.show()


## =====================================================
# TRANSFORMASI DATA KE FORMAT LONG (HOME + AWAY)
# =====================================================

# -----------------------------
# DATA TIM KANDANG (HOME)
# -----------------------------

# Memilih kolom yang relevan untuk tim kandang:
# - Date        : tanggal pertandingan
# - HomeTeam    : nama tim kandang
# - FTH Goals   : jumlah gol tim kandang (Goals For)
# - FTA Goals   : jumlah gol tim tandang (Goals Against)
# - FT Result   : hasil akhir pertandingan
home = df[[
    "Date",
    "HomeTeam",
    "FTH Goals",
    "FTA Goals",
    "FT Result"
]].copy()  # copy() digunakan untuk menghindari SettingWithCopyWarning

# Mengganti nama kolom agar konsisten dan mudah dianalisis
# Team  → nama tim
# GF    → Goals For (gol yang dicetak tim)
# GA    → Goals Against (gol yang kebobolan)
# Result → hasil pertandingan dari perspektif pertandingan
home.columns = ["Date", "Team", "GF", "GA", "Result"]

# Menambahkan indikator bahwa baris ini berasal dari tim kandang
# 1 = Home, 0 = Away
home["is_home"] = 1


# -----------------------------
# DATA TIM TANDANG (AWAY)
# -----------------------------

# Memilih kolom yang relevan untuk tim tandang
# Perhatikan bahwa posisi gol dibalik:
# - FTA Goals → GF (gol tim tandang)
# - FTH Goals → GA (gol yang kebobolan)
away = df[[
    "Date",
    "AwayTeam",
    "FTA Goals",
    "FTH Goals",
    "FT Result"
]].copy()

# Menyeragamkan nama kolom dengan dataset home
away.columns = ["Date", "Team", "GF", "GA", "Result"]

# Menambahkan indikator bahwa baris ini berasal dari tim tandang
away["is_home"] = 0


# -----------------------------
# GABUNGKAN DATA HOME & AWAY
# -----------------------------

# Menggabungkan dataset home dan away ke dalam satu DataFrame
# sehingga setiap baris merepresentasikan performa satu tim
# dalam satu pertandingan
long_df = pd.concat([home, away])

# Mengurutkan data berdasarkan tanggal pertandingan
# Hal ini penting untuk analisis time-series atau rolling statistics
long_df = long_df.sort_values("Date")

# Mereset index agar tetap berurutan setelah proses penggabungan dan pengurutan
long_df = long_df.reset_index(drop=True)

# =====================================================
# PERHITUNGAN POIN PERTANDINGAN (POINTS CALCULATION)
# =====================================================

# Fungsi untuk menghitung poin yang diperoleh sebuah tim
# berdasarkan hasil akhir pertandingan dan status kandang/tandang
def calc_points(row):
    """
    Menghitung poin pertandingan untuk satu tim.
    
    Aturan poin (standar liga sepak bola):
    - Menang  → 3 poin
    - Seri    → 1 poin
    - Kalah   → 0 poin

    Parameter:
    row : pandas.Series
        Satu baris data yang merepresentasikan satu tim
        dalam satu pertandingan (format long).

    Return:
    int
        Jumlah poin yang diperoleh tim pada pertandingan tersebut.
    """

    # Jika hasil pertandingan seri (Draw),
    # baik tim kandang maupun tandang mendapatkan 1 poin
    if row["Result"] == "D":
        return 1

    # Jika tim bermain sebagai kandang (is_home = 1)
    # dan hasil pertandingan adalah Home Win (H),
    # maka tim kandang menang dan memperoleh 3 poin
    if row["is_home"] == 1 and row["Result"] == "H":
        return 3

    # Jika tim bermain sebagai tandang (is_home = 0)
    # dan hasil pertandingan adalah Away Win (A),
    # maka tim tandang menang dan memperoleh 3 poin
    if row["is_home"] == 0 and row["Result"] == "A":
        return 3

    # Kondisi selain di atas berarti tim kalah
    # sehingga tidak memperoleh poin
    return 0


# Menerapkan fungsi calc_points ke setiap baris DataFrame long_df
# axis=1 menandakan fungsi dijalankan per baris
# Hasil perhitungan disimpan pada kolom baru bernama "Points"
long_df["Points"] = long_df.apply(calc_points, axis=1)


# =====================================================
# PEMBENTUKAN FITUR ROLLING (5 PERTANDINGAN TERAKHIR)
# =====================================================

# Mengurutkan data berdasarkan nama tim dan tanggal pertandingan
# Langkah ini WAJIB dilakukan sebelum rolling,
# agar perhitungan dilakukan secara kronologis untuk setiap tim
long_df = long_df.sort_values(["Team", "Date"])


# -----------------------------------------------------
# FORM TIM (RATA-RATA POIN 5 MATCH TERAKHIR)
# -----------------------------------------------------

# Menghitung rata-rata poin dari 5 pertandingan terakhir
# untuk setiap tim secara terpisah
#
# groupby("Team") → rolling dihitung per tim
# rolling(5)      → jendela 5 pertandingan terakhir
# min_periods=1   → jika tim belum punya 5 match,
#                   tetap dihitung menggunakan data yang tersedia
long_df["form"] = (
    long_df
    .groupby("Team")["Points"]
    .transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
)


# -----------------------------------------------------
# RATA-RATA GOALS FOR (GF) – 5 MATCH TERAKHIR
# -----------------------------------------------------

# Menghitung rata-rata jumlah gol yang dicetak tim
# dalam 5 pertandingan terakhir
long_df["gf_avg"] = (
    long_df
    .groupby("Team")["GF"]
    .transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
)


# -----------------------------------------------------
# RATA-RATA GOALS AGAINST (GA) – 5 MATCH TERAKHIR
# -----------------------------------------------------

# Menghitung rata-rata jumlah gol yang kebobolan tim
# dalam 5 pertandingan terakhir
long_df["ga_avg"] = (
    long_df
    .groupby("Team")["GA"]
    .transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
)


# =====================================================
# MENGGABUNGKAN (MERGE) FITUR TIM KEMBALI KE DATA MATCH
# =====================================================

# -----------------------------------------------------
# FITUR TIM KANDANG (HOME FEATURES)
# -----------------------------------------------------

# Memilih baris dari long_df yang merepresentasikan tim kandang
# serta hanya mengambil fitur-fitur yang relevan untuk modeling
home_feat = long_df[long_df["is_home"] == 1][
    ["Date", "Team", "form", "gf_avg", "ga_avg"]
]

# -----------------------------------------------------
# FITUR TIM TANDANG (AWAY FEATURES)
# -----------------------------------------------------

# Memilih baris dari long_df yang merepresentasikan tim tandang
# serta hanya mengambil fitur-fitur yang relevan untuk modeling
away_feat = long_df[long_df["is_home"] == 0][
    ["Date", "Team", "form", "gf_avg", "ga_avg"]
]

# -----------------------------------------------------
# MERGE FITUR HOME KE DATA MATCH UTAMA
# -----------------------------------------------------

# Menggabungkan fitur tim kandang ke DataFrame pertandingan (df)
# - left_on  → kunci join berasal dari df (Date + HomeTeam)
# - right_on → kunci join berasal dari home_feat (Date + Team)
# - how="left" → mempertahankan seluruh data match,
#                meskipun fitur tidak ditemukan (NaN)
df = df.merge(
    home_feat,
    left_on=["Date", "HomeTeam"],
    right_on=["Date", "Team"],
    how="left"
)

# -----------------------------------------------------
# MERGE FITUR AWAY KE DATA MATCH UTAMA
# -----------------------------------------------------

# Menggabungkan fitur tim tandang ke DataFrame pertandingan (df)
# - suffixes digunakan untuk membedakan fitur home dan away
#   setelah proses merge
df = df.merge(
    away_feat,
    left_on=["Date", "AwayTeam"],
    right_on=["Date", "Team"],
    how="left",
    suffixes=("_home", "_away")
)

# =====================================================
# PEMILIHAN FITUR DAN PEMBENTUKAN DATASET MODEL
# =====================================================

# Daftar fitur numerik yang akan digunakan sebagai input model
# Fitur-fitur ini merepresentasikan performa historis tim
# baik kandang maupun tandang (berdasarkan rolling 5 pertandingan terakhir)
FEATURES = [
    "form_home",     # Rata-rata poin 5 match terakhir tim kandang
    "form_away",     # Rata-rata poin 5 match terakhir tim tandang
    "gf_avg_home",   # Rata-rata gol dicetak tim kandang (5 match terakhir)
    "gf_avg_away",   # Rata-rata gol dicetak tim tandang (5 match terakhir)
    "ga_avg_home",   # Rata-rata gol kebobolan tim kandang (5 match terakhir)
    "ga_avg_away"    # Rata-rata gol kebobolan tim tandang (5 match terakhir)
]

# Menghapus baris yang memiliki nilai kosong (NaN)
# baik pada fitur maupun pada variabel target
# Hal ini penting agar model tidak menerima input yang tidak lengkap
df_model = df.dropna(subset=FEATURES + ["target"])


# -----------------------------------------------------
# PEMBENTUKAN INPUT (X) DAN TARGET (y)
# -----------------------------------------------------

# Matriks fitur (X) yang akan digunakan untuk training dan evaluasi model
X = df_model[FEATURES]

# Target klasifikasi:
# hasil akhir pertandingan yang telah di-encode
# (misal: A=0, D=1, H=2)
y = df_model["target"]

# -----------------------------------------------------
# TARGET TAMBAHAN UNTUK REGRESI GOL (OPSIONAL)
# -----------------------------------------------------

# Target regresi untuk memprediksi jumlah gol tim kandang
y_home_goals = df_model["FTH Goals"]

# Target regresi untuk memprediksi jumlah gol tim tandang
y_away_goals = df_model["FTA Goals"]


# =====================================================
# PEMBAGIAN DATA BERBASIS WAKTU (TIME-BASED SPLIT)
# =====================================================

# Menentukan indeks pemisah data training dan testing
# 80% data awal digunakan untuk training
# 20% data terakhir digunakan untuk testing
# Pendekatan ini menjaga urutan waktu sehingga
# tidak terjadi data leakage dari masa depan ke masa lalu
split_idx = int(len(df_model) * 0.8)

# -----------------------------
# DATA FITUR (X)
# -----------------------------

# Data training: pertandingan-pertandingan awal (historical data)
X_train = X.iloc[:split_idx]

# Data testing: pertandingan-pertandingan terbaru
X_test  = X.iloc[split_idx:]


# -----------------------------
# TARGET KLASIFIKASI
# -----------------------------

# Target hasil pertandingan untuk data training
y_train = y.iloc[:split_idx]

# Target hasil pertandingan untuk data testing
y_test  = y.iloc[split_idx:]


# -----------------------------
# TARGET REGRESI GOL (HOME & AWAY)
# -----------------------------

# Target jumlah gol tim kandang (training)
y_home_train = y_home_goals.iloc[:split_idx]

# Target jumlah gol tim kandang (testing)
y_home_test = y_home_goals.iloc[split_idx:]

# Target jumlah gol tim tandang (training)
y_away_train = y_away_goals.iloc[:split_idx]

# Target jumlah gol tim tandang (testing)
y_away_test = y_away_goals.iloc[split_idx:]


# =====================================================
# NORMALISASI FITUR (SCALING) – KHUSUS UNTUK MODEL KNN
# =====================================================

# Membuat objek StandardScaler
# StandardScaler melakukan transformasi:
# (x - mean) / standard deviation
# sehingga setiap fitur memiliki mean = 0 dan std = 1
scaler = StandardScaler()

# Melakukan fitting scaler HANYA pada data training
# untuk mencegah kebocoran informasi dari data testing
X_train_s = scaler.fit_transform(X_train)

# Menerapkan scaler yang sama ke data testing
# tanpa melakukan fit ulang
X_test_s  = scaler.transform(X_test)

# =====================================================
# PEMODELAN KLASIFIKASI HASIL PERTANDINGAN
# =====================================================

# -----------------------------------------------------
# RANDOM FOREST CLASSIFIER
# -----------------------------------------------------

# Membuat model Random Forest
# Random Forest adalah ensemble learning berbasis decision tree
# yang mampu menangkap hubungan non-linear antar fitur
rf = RandomForestClassifier(
    n_estimators=300,   # Jumlah pohon keputusan dalam forest
    max_depth=12,      # Kedalaman maksimum setiap pohon
    random_state=42    # Seed untuk memastikan hasil eksperimen reproducible
)

# Melatih model Random Forest menggunakan data training
rf.fit(X_train, y_train)

# Melakukan prediksi hasil pertandingan pada data testing
rf_pred = rf.predict(X_test)


# -----------------------------------------------------
# XGBOOST CLASSIFIER
# -----------------------------------------------------

# Membuat model XGBoost untuk klasifikasi multi-kelas
# XGBoost menggunakan teknik gradient boosting
# yang sangat efektif untuk data tabular
xgb = XGBClassifier(
    n_estimators=300,          # Jumlah boosting rounds (trees)
    learning_rate=0.05,        # Learning rate untuk memperlambat proses learning
    max_depth=6,               # Kedalaman maksimum tree
    subsample=0.9,             # Proporsi data yang digunakan per tree
    colsample_bytree=0.9,      # Proporsi fitur yang digunakan per tree
    objective="multi:softprob",# Objective untuk multi-class classification
    num_class=3,               # Jumlah kelas target (A, D, H)
    random_state=42,           # Seed agar hasil konsisten
    eval_metric="mlogloss"     # Metric evaluasi internal selama training
)

# Melatih model XGBoost menggunakan data training
xgb.fit(X_train, y_train)

# Melakukan prediksi kelas pada data testing
xgb_pred = xgb.predict(X_test)


# -----------------------------------------------------
# K-NEAREST NEIGHBORS (KNN)
# -----------------------------------------------------

# Membuat model KNN
# KNN adalah algoritma berbasis jarak,
# sehingga sangat sensitif terhadap skala fitur
knn = KNeighborsClassifier(
    n_neighbors=15  # Jumlah tetangga terdekat yang digunakan
)

# Melatih model KNN menggunakan data training yang SUDAH diskalakan
knn.fit(X_train_s, y_train)

# Melakukan prediksi menggunakan data testing yang SUDAH diskalakan
knn_pred = knn.predict(X_test_s)


# =====================================================
# MODEL REGRESI UNTUK PREDIKSI SKOR PERTANDINGAN
# =====================================================

print("\n=== TRAINING SCORE PREDICTION MODELS ===")


# -----------------------------------------------------
# MODEL REGRESI: PREDIKSI GOL TIM KANDANG (HOME GOALS)
# -----------------------------------------------------

# Membuat model XGBoost Regressor untuk memprediksi
# jumlah gol yang dicetak oleh tim kandang
xgb_home_score = XGBRegressor(
    n_estimators=300,     # Jumlah pohon (boosting rounds)
    learning_rate=0.05,   # Learning rate untuk pembelajaran bertahap
    max_depth=6,          # Kedalaman maksimum tiap pohon
    subsample=0.9,        # Proporsi data yang digunakan per tree
    colsample_bytree=0.9,# Proporsi fitur yang digunakan per tree
    random_state=42       # Seed agar hasil eksperimen konsisten
)

# Melatih model menggunakan data training
xgb_home_score.fit(X_train, y_home_train)

# Melakukan prediksi jumlah gol tim kandang pada data testing
home_pred = xgb_home_score.predict(X_test)


# -----------------------------------------------------
# MODEL REGRESI: PREDIKSI GOL TIM TANDANG (AWAY GOALS)
# -----------------------------------------------------

# Membuat model XGBoost Regressor untuk memprediksi
# jumlah gol yang dicetak oleh tim tandang
xgb_away_score = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

# Melatih model menggunakan data training
xgb_away_score.fit(X_train, y_away_train)

# Melakukan prediksi jumlah gol tim tandang pada data testing
away_pred = xgb_away_score.predict(X_test)


# =====================================================
# EVALUASI MODEL PREDIKSI SKOR
# =====================================================

# -----------------------------------------------------
# MEAN ABSOLUTE ERROR (MAE)
# -----------------------------------------------------

# Menghitung MAE untuk prediksi gol tim kandang
home_mae = mean_absolute_error(y_home_test, home_pred)

# Menghitung MAE untuk prediksi gol tim tandang
away_mae = mean_absolute_error(y_away_test, away_pred)


# -----------------------------------------------------
# EXACT SCORE EVALUATION (SETELAH PEMBULATAN)
# -----------------------------------------------------

# Membulatkan hasil prediksi ke bilangan bulat terdekat
# karena skor pertandingan bersifat diskrit
home_pred_rounded = np.round(home_pred)
away_pred_rounded = np.round(away_pred)

# Menghitung akurasi prediksi skor tepat (exact match)
# secara terpisah untuk tim kandang dan tandang
home_exact_acc = accuracy_score(y_home_test, home_pred_rounded)
away_exact_acc = accuracy_score(y_away_test, away_pred_rounded)


# -----------------------------------------------------
# AKURASI SKOR PERTANDINGAN LENGKAP
# -----------------------------------------------------

# Exact score accuracy:
# prediksi dianggap benar jika skor home DAN away
# keduanya tepat secara bersamaan
exact_match = (
    (home_pred_rounded == y_home_test.values) &
    (away_pred_rounded == y_away_test.values)
)
exact_score_acc = exact_match.sum() / len(y_home_test)


# -----------------------------------------------------
# GOAL DIFFERENCE ACCURACY
# -----------------------------------------------------

# Menghitung selisih gol prediksi dan aktual
pred_diff = home_pred_rounded - away_pred_rounded
actual_diff = y_home_test.values - y_away_test.values

# Mengevaluasi apakah arah hasil pertandingan
# (menang, seri, kalah) berhasil diprediksi
diff_acc = accuracy_score(
    np.sign(actual_diff),
    np.sign(pred_diff)
)


# -----------------------------------------------------
# WITHIN ±1 GOAL ACCURACY
# -----------------------------------------------------

# Mengecek apakah prediksi gol home berada dalam selisih ±1 gol
home_within_1 = np.abs(home_pred_rounded - y_home_test.values) <= 1

# Mengecek apakah prediksi gol away berada dalam selisih ±1 gol
away_within_1 = np.abs(away_pred_rounded - y_away_test.values) <= 1

# Akurasi dihitung jika kedua prediksi (home & away)
# berada dalam toleransi ±1 gol
within_1_acc = (
    home_within_1 & away_within_1
).sum() / len(y_home_test)


# =====================================================
# LAPORAN PERFORMA PREDIKSI SKOR PERTANDINGAN
# =====================================================

# Header laporan agar output mudah dibaca di console
print("\n" + "=" * 60)
print("SCORE PREDICTION PERFORMANCE REPORT")
print("=" * 60)


# -----------------------------------------------------
# MEAN ABSOLUTE ERROR (MAE)
# -----------------------------------------------------

# MAE mengukur rata-rata selisih absolut antara
# prediksi dan nilai aktual (semakin kecil semakin baik)
print(f"\n📊 Mean Absolute Error (MAE):")
print(f"   Home Goals MAE: {home_mae:.3f}")
print(f"   Away Goals MAE: {away_mae:.3f}")

# Rata-rata MAE dari prediksi gol home dan away
# Digunakan sebagai indikator error keseluruhan model skor
print(f"   Average MAE   : {(home_mae + away_mae) / 2:.3f}")


# -----------------------------------------------------
# EXACT GOAL ACCURACY
# -----------------------------------------------------

# Mengukur seberapa sering model memprediksi
# jumlah gol secara tepat (setelah pembulatan)
print(f"\n🎯 Exact Goal Accuracy:")
print(f"   Home Goals Exact: {home_exact_acc * 100:.2f}%")
print(f"   Away Goals Exact: {away_exact_acc * 100:.2f}%")


# -----------------------------------------------------
# EXACT SCORE MATCH
# -----------------------------------------------------

# Evaluasi paling ketat:
# prediksi dianggap benar hanya jika skor home
# DAN skor away sama persis dengan skor aktual
print(f"\n⚽ Exact Score Match:")
print(f"   Both scores correct: {exact_score_acc * 100:.2f}%")


# -----------------------------------------------------
# GOAL DIFFERENCE ACCURACY
# -----------------------------------------------------

# Mengukur apakah model berhasil memprediksi
# arah hasil pertandingan:
# menang kandang, seri, atau menang tandang
print(f"\n📈 Goal Difference Accuracy:")
print(f"   Correct winner/draw: {diff_acc * 100:.2f}%")


# -----------------------------------------------------
# WITHIN ±1 GOAL ACCURACY
# -----------------------------------------------------

# Mengukur toleransi prediksi:
# prediksi dianggap baik jika selisih gol
# tidak lebih dari 1 untuk kedua tim
print(f"\n✅ Within 1 Goal Accuracy:")
print(f"   Both within ±1 goal: {within_1_acc * 100:.2f}%")


# -----------------------------------------------------
# DISTRIBUSI ERROR PREDIKSI
# -----------------------------------------------------

# Menghitung error absolut antara prediksi (rounded)
# dan skor aktual untuk analisis lebih mendalam
print(f"\n📉 Error Distribution:")

home_errors = np.abs(home_pred_rounded - y_home_test.values)
away_errors = np.abs(away_pred_rounded - y_away_test.values)

# Distribusi error untuk tim kandang
print(f"   Home - 0 error : {(home_errors == 0).sum() / len(home_errors) * 100:.1f}%")
print(f"   Home - 1 error : {(home_errors == 1).sum() / len(home_errors) * 100:.1f}%")
print(f"   Home - 2+ error: {(home_errors >= 2).sum() / len(home_errors) * 100:.1f}%")

# Distribusi error untuk tim tandang
print(f"   Away - 0 error : {(away_errors == 0).sum() / len(away_errors) * 100:.1f}%")
print(f"   Away - 1 error : {(away_errors == 1).sum() / len(away_errors) * 100:.1f}%")
print(f"   Away - 2+ error: {(away_errors >= 2).sum() / len(away_errors) * 100:.1f}%")

# Footer laporan
print("=" * 60)

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
        
        
        print("\n" + "="*60)
        lanjut = input("\nPrediksi pertandingan lain? (y/n): ").strip().lower()
        if lanjut != 'y':
            print("\nTerima kasih! Program selesai.")
            break
        
        # =====================================================
# SAVE MODEL & SCALER
# =====================================================

# =====================================================
# SAVE TRAINED MODELS & ARTIFACTS
# =====================================================

# =====================================================
# SAVE TRAINED MODELS & ARTIFACTS (LENGKAP & AMAN)
# =====================================================

SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# CLASSIFICATION MODELS
# =====================
joblib.dump(rf, os.path.join(SAVE_DIR, "rf_clf.pkl"))
joblib.dump(xgb, os.path.join(SAVE_DIR, "xgb_clf.pkl"))
joblib.dump(knn, os.path.join(SAVE_DIR, "knn_clf.pkl"))

# =====================
# SCORE REGRESSION MODELS
# =====================
joblib.dump(xgb_home_score, os.path.join(SAVE_DIR, "xgb_goal_home.pkl"))
joblib.dump(xgb_away_score, os.path.join(SAVE_DIR, "xgb_goal_away.pkl"))

# =====================
# PREPROCESSING OBJECTS
# =====================
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(SAVE_DIR, "target_le.pkl"))

# =====================
# FEATURE METADATA
# =====================
joblib.dump(FEATURES, os.path.join(SAVE_DIR, "feature_cols.pkl"))

print("✅ Semua model & artefak berhasil disimpan ke:")
print(SAVE_DIR)



# =====================================================
# JALANKAN CLI
# =====================================================
if __name__ == "__main__":
    main_cli()