# Deteksi URL Phishing (Defensive AI)

Final Project Keamanan Informasi - Universitas Siliwangi.
Klasifikasi URL phishing vs legitimate memakai fitur leksikal (tanpa membuka situs) dengan
Logistic Regression, Random Forest, dan XGBoost.

## Struktur
| File | Fungsi |
|---|---|
| `features.py` | Ekstraksi 29 fitur leksikal dari string URL |
| `train.py` | Training, perbandingan model, evaluasi, analisis FP/FN |
| `app.py` | Demo web Streamlit (1 URL atau batch CSV) |
| `make_sample_data.py` | Dataset SINTETIS untuk smoke test saja |

## Cara menjalankan
```bash
pip install -r requirements.txt

# 1. Unduh dataset PhiUSIIL (UCI ML Repository / Kaggle), taruh di folder data/
# 2. Training (di PhiUSIIL label 1 = legitimate, jadi --legit-label 1)
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --legit-label 1

# 3. Jalankan demo
streamlit run app.py
```
Opsi: `--sample 50000` untuk mempercepat, `--url-col` / `--label-col` bila nama kolom berbeda.
Untuk dataset lain, atur `--legit-label` sesuai nilai label URL yang aman.

## Output (folder `reports/`)
- `metrics.json`, `model_comparison.csv`: precision, recall, F1 tiap model
- `confusion_matrix.png`, `feature_importance.png`
- `false_positives.csv`, `false_negatives.csv`: bahan analisis FP/FN di laporan

## Catatan metodologi (bagus untuk bagian pembahasan)
- URL duplikat dihapus, split train/test dilakukan per hostname agar domain yang sama tidak bocor ke test.
- Hasil di PhiUSIIL cenderung sangat tinggi karena dataset relatif mudah dan ada pola artefak
  (misalnya `https`). Bahas ini sebagai keterbatasan, dan uji juga pada data lain jika sempat.
- Model hanya melihat teks URL; tidak mengecek konten halaman, usia domain, atau reputasi.
