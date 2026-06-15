# Container image for the AURA prediction API (production serving).
FROM python:3.12-slim

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29"

# App code + model artifacts (ensure Git LFS objects are pulled before build).
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
