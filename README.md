# ESCI Product Relevance Classification

Perbandingan pendekatan **lexical** (TF-IDF + Linear SVM) vs **semantic**
(DeBERTa-v3-base) untuk klasifikasi relevansi query-produk 4 kelas (E/S/C/I)
pada Amazon Shopping Queries **ESCI** dataset.

Judul skripsi: *Perbandingan Pendekatan Lexical dan Semantic pada Product
Search Menggunakan Amazon Shopping Queries ESCI Dataset: Analisis Kinerja dan
Karakteristik Hasil Klasifikasi*.

## Ruang lingkup yang dikunci (per arahan pembimbing)

- **Data**: hanya `product_locale == "us"` dan `large_version == 1`.
- **Fitur teks**: hanya `query` dan `product_title`. `product_description`
  dan `product_bullet_point` **tidak pernah dibaca atau dibersihkan**
  di mana pun dalam pipeline ini (lihat `src/data/clean.py::TEXT_COLUMNS`).
- **Model semantic**: hanya **DeBERTa-v3-base**. BERT-base dan RoBERTa-base
  sempat dicoba saat development tapi **dibuang** sesuai catatan pembimbing —
  tidak ada sisa kode/konfigurasi untuk keduanya di repo ini.
- **Split resmi**: kolom `split` bawaan dataset dipakai apa adanya (tidak ada
  split manual ulang).
  - 100.000 baris *stratified* dari official **TRAIN** → dipecah tetap
    menjadi **90.000 training** + **10.000 validation**.
  - 30.551 baris *stratified* dari official **TEST** → dipakai **hanya**
    untuk evaluasi akhir (`--final-evaluation`), tidak pernah untuk
    development/model selection/tuning/early stopping.
  - Seed: `42` di semua tahap (sampling, split, training).

## Struktur repo

```
configs/
  main_experiment.yaml        # 2 lexical + 1 semantic (deberta) experiment
  pilot_lexical.yaml          # sanity-check kecil, opsional
  pilot_semantic_deberta.yaml # sanity-check kecil, opsional
scripts/
  build_us_representative_subset.py   # jalan SEKALI, membuat 4 file parquet + report
  run_pilot.py                        # opsional: pilot DeBERTa skala kecil
  run_main_experiment.py              # jalan per-experiment (dev / --final-evaluation)
  sanity_check_no_leakage.py          # audit independen: no-overlap, no-leakage
src/
  data/    load.py, clean.py, sample.py, split.py
  features/tfidf.py
  models/  lexical_svm.py, semantic_transformer.py
  evaluation/metrics.py
  utils/   config.py, seed.py
notebooks/
  ESCI_Sempro_Runner.ipynb    # notebook siap-pakai untuk Colab + sudah lengkap markdown untuk sempro
```

## Alur (tanpa leakage)

```
raw parquet (examples + products)
        |
  load_us_esci()                    -> filter locale=us, large_version=1
        |
  get_official_split()              -> official TRAIN / official TEST (kolom `split`)
        |                                   |
  stratified_sample(n=100000)        stratified_sample(n=30551)
        |                                   |
  stratified_train_validation_split         |
        |                                   |
  training (90k) + validation (10k)   test (30.551, DIKUNCI, hanya utk final eval)
```

- **TF-IDF** di-`fit` **hanya** pada teks *training* 90K; validation & test
  cuma di-`transform` (`src/features/tfidf.py`, ditegakkan lagi di
  `scripts/run_main_experiment.py::_run_lexical`).
- **DeBERTa** di-fine-tune **hanya** pada training 90K; validation 10K dipakai
  untuk `eval_loss`, checkpointing, dan early stopping. Test **tidak pernah**
  dibaca kecuali skrip dipanggil dengan `--final-evaluation`.
- `scripts/run_main_experiment.py` memvalidasi jumlah baris tiap split
  (`training_rows`, `validation_rows`, `test_subset_rows`) dan melakukan
  pengecekan `example_id` yang tidak boleh overlap antar split sebelum
  training dimulai — jika ada overlap, script langsung error, bukan diam-diam
  lanjut.
- `scripts/sanity_check_no_leakage.py` adalah audit tambahan yang independen
  dari kode training — jalankan ini sekali sebelum sidang untuk punya bukti
  konkret (output tercetak) bahwa tidak ada kebocoran data.

## Cara pakai (lokal)

```bash
pip install -r requirements.txt

# 1) Sekali saja: bangun 4 split tetap dari raw parquet
python -m scripts.build_us_representative_subset \
    --raw-data-dir /path/to/raw \
    --output-dir /path/to/processed \
    --seed 42

# 2) Audit no-leakage (opsional tapi disarankan sebelum sidang)
python -m scripts.sanity_check_no_leakage --data-dir /path/to/processed

# 3) (opsional) pilot DeBERTa skala kecil, sanity-check pipeline training
python -m scripts.run_pilot \
    --config configs/pilot_semantic_deberta.yaml \
    --data-dir /path/to/raw \
    --output-dir /path/to/pilot_outputs

# 4) Development (baca training 90K + validation 10K saja)
python -m scripts.run_main_experiment --experiment lexical_separate     --data-dir /path/to/processed --output-dir /path/to/outputs
python -m scripts.run_main_experiment --experiment lexical_concatenated --data-dir /path/to/processed --output-dir /path/to/outputs
python -m scripts.run_main_experiment --experiment semantic_deberta     --data-dir /path/to/processed --output-dir /path/to/outputs

# 5) Final evaluation (baru baca test 30.551, HANYA setelah semua keputusan dikunci)
python -m scripts.run_main_experiment --experiment lexical_separate     --data-dir /path/to/processed --output-dir /path/to/outputs --final-evaluation
python -m scripts.run_main_experiment --experiment lexical_concatenated --data-dir /path/to/processed --output-dir /path/to/outputs --final-evaluation
python -m scripts.run_main_experiment --experiment semantic_deberta     --data-dir /path/to/processed --output-dir /path/to/outputs --final-evaluation
```

## Cara pakai (Google Colab)

Buka `notebooks/ESCI_Sempro_Runner.ipynb` di Colab. Notebook ini:

1. Mount Google Drive.
2. Clone/pull repo ini dari GitHub (ganti `REPOSITORY_URL` di cell konfigurasi
   kalau kamu push ke repo/branch lain).
3. Install `requirements.txt`.
4. Menjalankan langkah 1-5 di atas sebagai cell terpisah, plus menampilkan
   tabel audit ukuran/distribusi label, tabel validation results, dan tabel
   final test metrics + confusion matrix per model.
5. Training DeBERTa mendukung **resume otomatis** dari checkpoint terakhir
   yang *lengkap* di Drive (dicek lewat keberadaan `trainer_state.json`),
   jadi aman kalau runtime Colab terputus di tengah training.

## Interpretasi hasil

- Metrik utama: **Macro-F1** dan **Weighted-F1**. Metrik tambahan: Accuracy,
  Micro-F1.
- Model selection (development) memakai hasil **validation** saja.
- Angka final (untuk Bab 4) memakai hasil **final_test** — hanya diisi
  setelah `--final-evaluation` dijalankan.
- `per_class_f1` dan `confusion_matrix` dipakai untuk membandingkan pola
  kesalahan lexical vs semantic (mis. apakah semantic lebih baik
  membedakan Substitute vs Complement, dsb.) — ini karakterisasi hasil,
  bukan klaim sebab-akibat.

## Catatan perbaikan dari draft kode sebelumnya

- `semantic_transformer.py` sebelumnya terpotong/tidak lengkap — sekarang
  full implementasi dengan Hugging Face `Trainer`, `EarlyStoppingCallback`
  (berbasis `eval_loss`), resume-from-checkpoint yang aman (hanya resume dari
  checkpoint yang punya `trainer_state.json`, supaya tidak resume dari
  checkpoint yang korup akibat runtime terputus), dan riwayat loss yang
  ikut disimpan ke `results.json`.
- `configs/main_experiment.yaml` sebelumnya masih memuat `semantic_bert` dan
  `pilot_semantic_bert_5k.yaml` / `pilot_semantic_roberta.yaml` — semua
  dihapus; hanya `semantic_deberta` yang tersisa.
- `src/utils/config.py` dan `src/utils/seed.py` sebelumnya dirujuk oleh
  script tapi belum ada isinya — sekarang lengkap.
- Query untuk arm lexical sebelumnya **tidak** ikut dinormalisasi (lowercase
  dsb.) — hanya `product_title` yang dinormalisasi, sementara `query` dipakai
  mentah. Ini sudah diperbaiki: `query` sekarang juga melalui
  `_lexical_normalize_one` / `_semantic_normalize_one` yang sama, supaya
  perlakuan kedua sisi teks konsisten.
- Ditambahkan `scripts/sanity_check_no_leakage.py` sebagai bukti tambahan
  untuk sidang bahwa training/validation/test tidak overlap.
