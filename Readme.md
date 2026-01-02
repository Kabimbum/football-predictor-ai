# Langkah Instalasi Proyek Prediksi Sepak Bola

## 1. Clone Repository
Clone repository ke komputer Anda:


## 2. Buat Virtual Environment (Opsional)
Disarankan menggunakan virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Buat file bernama requirements.txt

```bash
flask
pandas
scikit-learn
xgboost
joblib
scipy
numpy
```

Selanjutnya di terminal:

```bash
pip install -r requirements.txt
```

## 4. Siapkan Data
Pastikan file data sudah tersedia di:

## 5. Training Model
Jalankan script training untuk melatih dan menyimpan model:
```bash
python prediksi/training_model.py
```

## 6. Jalankan Aplikasi Web
Jalankan aplikasi Flask:
```bash
python prediksi/app.py
```

## 7. Akses Aplikasi
Buka browser dan akses:
```
http://localhost:5000
```

---

**Catatan:**  
- Pastikan semua file model (`model_xgb.pkl`, dll) sudah dihasilkan sebelum menjalankan aplikasi web.
- Jika ada error, cek kembali dependensi dan path data.