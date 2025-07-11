# Stage 1: Base Image
# Use a slim Python image based on Debian Bullseye for 'apt' availability.
FROM python:3.9-slim-bullseye

# Set environment variables to prevent Python from writing .pyc files and to
# ensure output is sent straight to the terminal without being buffered.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Stage 2: Install System Dependencies
# Update package lists, install Tesseract OCR with all available language packs,
# and then clean up apt lists to keep the final image size smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 3: Application Setup
# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if the requirements file changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container.
COPY . .

