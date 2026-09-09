# Forecasting Jumlah Penumpang TransJakarta

Project ini memprediksi jumlah penumpang TransJakarta per hari menggunakan pendekatan machine learning berbasis data historis. Model utama yang digunakan adalah **Random Forest Regressor**, dengan fitur yang dibangun dari pola historis jumlah penumpang (lag & rolling statistics) serta informasi hari dalam seminggu.

## Dataset

Data bersumber dari Satu Data Jakarta — jumlah penumpang angkutan umum yang terlayani per hari, difilter khusus untuk moda `transjakarta`.

Link dataset: https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=jumlah-penumpang-angkutan-umum-yang-terlayani-perhari&data_no=1

## Alur Project

1. **Import Library** — pandas, numpy, matplotlib, scikit-learn (Linear Regression, Random Forest, GridSearchCV, TimeSeriesSplit).
2. **Load & EDA** — memeriksa struktur data, periode, dan jenis moda transportasi yang tersedia.
3. **Preprocessing** — filter data ke moda `transjakarta`, konversi tipe data, penanganan missing value pada target, dan pengurutan berdasarkan waktu.
4. **Feature Engineering** — membuat fitur `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_std_7`, dan `day_of_week`.
5. **Train-Test Split** — pembagian 80:20 berbasis waktu (bukan random split) agar tidak terjadi kebocoran data dari masa depan.
6. **Modeling** — Linear Regression sebagai baseline, dilanjutkan dengan Random Forest Regressor sebagai model utama.
7. **Hyperparameter Tuning** — GridSearchCV dengan `TimeSeriesSplit` (5 fold) sebagai skema cross-validation, dioptimasi terhadap MAE.
8. **Evaluasi** — model final diuji pada test set yang terpisah, dilengkapi validasi lintas periode melalui time series cross-validation.

## Hasil

Model final (Random Forest setelah tuning) pada test set:

| Metrik | Nilai |
|---|---|
| MAE  | ± 78 ribu penumpang/hari |
| RMSE | ± 130 ribu penumpang/hari |
| R²   | ± 0.76 |

Model mampu menangkap sebagian besar pola variasi jumlah penumpang pada periode testing, meskipun hasil cross-validation menunjukkan performa dapat bervariasi antarperiode.

## Fitur yang Digunakan

- `lag_1`, `lag_7`, `lag_14`, `lag_28` — jumlah penumpang pada periode-periode sebelumnya
- `rolling_mean_7`, `rolling_std_7` — rata-rata dan standar deviasi bergulir 7 hari
- `day_of_week` — hari dalam seminggu

## Cara Menjalankan

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Letakkan dataset CSV di folder `data/data.csv`, lalu jalankan notebook:

```bash
jupyter notebook transjakarta_portfolio.ipynb
```

## Struktur Repo

```
.
├── transjakarta_portfolio.ipynb
├── data/
│   └── data.csv
└── README.md
```

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib
