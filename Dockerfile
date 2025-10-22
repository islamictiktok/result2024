FROM python:3.10-slim

# تثبيت Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# نسخ الملفات
WORKDIR /app
COPY . .

# تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل السكريبت
CMD ["python", "main.py"]
