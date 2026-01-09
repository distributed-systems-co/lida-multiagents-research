FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
