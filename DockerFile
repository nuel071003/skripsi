FROM python:3.10-slim

# Mencegah file .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Menampilkan log langsung
ENV PYTHONUNBUFFERED=1

# Port Hugging Face Space
ENV PORT=7860

WORKDIR /app

# Install dependency sistem
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install library Python
RUN pip install --no-cache-dir -r requirements.txt

# Port
EXPOSE 7860

# Jalankan Flask
CMD ["python", "app.py"]
