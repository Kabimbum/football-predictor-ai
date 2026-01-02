# Dokumentasi Model pada `training_model.py`

## 1. Model yang Digunakan

### A. XGBoost Classifier
- **Jenis Model**: Klasifikasi multikelas (Menang, Seri, Kalah).
- **Algoritma**: Gradient Boosting Machine berbasis pohon keputusan.
- **Fitur Input**: Statistik pertandingan seperti jumlah gol (gf), kebobolan (ga), expected goals (xg), penguasaan bola (poss), tembakan (sh), venue, kompetisi, kapten, dll.
- **Proses**:
  - Data kategorikal diubah menggunakan LabelEncoder.
  - Target hasil pertandingan di-mapping ke angka: Menang (2), Seri (1), Kalah (0).
  - Data dibersihkan dari nilai kosong dan diurutkan berdasarkan tanggal.
  - Data dibagi secara time-based (80% training, 20% testing).
  - Model XGBoost dilatih dengan parameter konservatif untuk menghindari overfitting.
  - Evaluasi menggunakan akurasi dan classification report.
  - Feature importance ditampilkan untuk mengetahui fitur paling berpengaruh.

### B. Poisson Score Predictor
- **Jenis Model**: Statistik prediktif berbasis distribusi Poisson.
- **Algoritma**: Menghitung rata-rata statistik serangan dan pertahanan tiap tim, lalu menggunakan distribusi Poisson untuk memperkirakan kemungkinan skor akhir.
- **Fitur Input**: Statistik rata-rata serangan dan pertahanan tiap tim (gf, ga, xg, xga).
- **Proses**:
  - Statistik tiap tim dihitung dari data historis.
  - Prediksi skor dilakukan dengan mengestimasi expected goals dan probabilitas skor menggunakan distribusi Poisson.
  - Evaluasi menggunakan Mean Absolute Error (MAE) antara skor prediksi dan skor aktual.
  - Menampilkan prediksi skor paling mungkin dan tiga skor teratas beserta probabilitasnya.

---

## 2. Proses Training dan Evaluasi

1. **Preprocessing Data**
   - Label encoding untuk fitur kategorikal.
   - Mapping hasil pertandingan ke angka.
   - Pembersihan data dari nilai kosong.

2. **Training Model**
   - XGBoost dilatih untuk klasifikasi hasil pertandingan.
   - PoissonScorePredictor menghitung statistik tim dan melakukan prediksi skor.

3. **Evaluasi Model**
   - XGBoost: Akurasi dan classification report.
   - Poisson: MAE (Mean Absolute Error).

4. **Penyimpanan Model**
   - Semua model dan objek penting disimpan menggunakan `joblib` untuk digunakan pada prediksi selanjutnya.

---

## 3. Output dan File yang Dihasilkan

- `model_xgb.pkl`: Model XGBoost terlatih.
- `label_encoders.pkl`: Encoder untuk fitur kategorikal.
- `result_mapping.pkl`: Mapping hasil pertandingan.
- `poisson_predictor.pkl`: Model prediksi skor Poisson.
- `features.pkl`: Daftar fitur yang digunakan.

---

## 4. Kesimpulan

File `training_model.py` menggunakan dua pendekatan:
- **XGBoost** untuk klasifikasi hasil pertandingan.
- **Poisson Score Predictor** untuk prediksi skor akhir pertandingan.

Kedua model ini saling melengkapi untuk analisis dan prediksi pertandingan sepak bola secara statistik dan machine learning.

---