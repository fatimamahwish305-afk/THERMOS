# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for pandas/numpy/xgboost/scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Copy requirements first to leverage docker cache
COPY core-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core-backend directory contents into the container at /app/core-backend
COPY core-backend /app/core-backend

# Expose port 10000
EXPOSE 10000

# Command to run the application
# We need to set the PYTHONPATH so that core-backend is recognized as a package
ENV PYTHONPATH=/app
CMD ["uvicorn", "core-backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
