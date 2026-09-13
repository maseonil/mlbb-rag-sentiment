FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy & install requirements langsung (tanpa kompilasi Debian)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code proyek
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]