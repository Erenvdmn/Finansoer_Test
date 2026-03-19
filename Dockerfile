# 1. Taban olarak Python sürümü seçiyoruz
FROM python:3.12-slim

# 2. Konteyner içindeki çalışma dizinimizi belirliyoruz
WORKDIR /app

# 3. Sistem kütüphanelerini güncelliyoruz (Yapay zeka paketleri için gerekli C++ derleyicileri vs.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Önce sadece requirements.txt dosyasını kopyalayıp kütüphaneleri kuruyoruz
# (Bu sayede kod değişse bile kütüphaneleri baştan indirmemiş oluruz, önbellek kullanır)
COPY requirements.txt .

# 5. Kütüphaneleri indir (PyTorch çok büyük olduğu için biraz sürebilir)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Kalan tüm kodlarımızı (main.py, modüllerimiz vs.) içeri kopyalıyoruz
COPY . .

# 7. FastAPI'nin çalışacağı portu dış dünyaya açıyoruz
EXPOSE 8000

# 8. Konteyner ayağa kalktığında çalışacak olan nihai komut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]