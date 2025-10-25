# Multi-Service Voice Recommendation System Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    curl \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements files FIRST (for better caching)
COPY requirements.txt /requirements.txt


# Install Python dependencies (this layer will be cached)
RUN pip install  --upgrade pip && \
    pip install  -r /requirements.txt
# Copy application code AFTER installing dependencies
COPY backend/ /app/backend/
COPY backend_transcription/ /app/backend_transcription/
COPY frontend/ /app/frontend/
COPY embedding_data/ /app/embedding_data/

# Create necessary directories
RUN mkdir -p /app/backend/uploads /app/backend/logs /app/backend_transcription/uploads

# Configure nginx for frontend
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-enabled/frontend.conf

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports
# 5020 - Backend API
# 8000 - Transcription API
# 80 - Frontend (nginx)
EXPOSE 5020 8000 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5020/health && curl -f http://localhost:8000/health || exit 1

# Run entrypoint script which applies fixes and starts supervisor
CMD ["/entrypoint.sh"]

