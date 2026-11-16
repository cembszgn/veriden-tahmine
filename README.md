# Veriden Tahmine — Diyabet Hastalığı İlerleme Skoru

## İçerik
- `veri_seti.csv`: 442 satır, 3 özellik ve 1 hedef
- `veri_sozlugu.csv`: sütun açıklamaları
- `veriden_tahmine_projesi.ipynb`: EDA, model, metrikler, 3 yeni örnek ve mini uygulama
- `model_ve_tahmin.py`: aynı modeli tekrar eğiten, metrikleri yazdıran ve terminalden kullanıcı girdisi alan betik
- `proje_raporu.pdf`, `proje_sunumu.pptx`, iki grafik ve `test_tahminleri.csv`

## Kurulum ve çalıştırma
Python 3.10+ ortamında: `pip install pandas scikit-learn matplotlib notebook`
Ardından proje klasöründe `python model_ve_tahmin.py` veya notebook'u Jupyter/Colab'da açın.

## Yöntem
Linear Regression; %80/%20 train-test; MAE, RMSE ve R². Veriler scikit-learn `load_diabetes` benchmark kaynağından gelir. Çalışma klinik karar aracı değildir.
