FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
