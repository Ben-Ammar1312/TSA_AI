# ===== Stage 1: Build dependencies =====
FROM python:3.11-slim AS builder
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python deps into a wheel cache
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ===== Stage 2: Runtime image =====
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime system deps
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy project
COPY . .

# Expose Django port
EXPOSE 8000

# Collect static files (optional if needed)
# RUN python manage.py collectstatic --noinput

# Start Django with Gunicorn
CMD ["gunicorn", "TSA_AI.wsgi:application", "--bind", "0.0.0.0:8000"]
