# KCC Voice Recommendation System

A containerized multi-service voice recommendation system with Docker Compose.

## Architecture

This application runs three services in a single Docker container:

1. **Backend API** (Port 5000) - Main FastAPI application for voice processing
2. **Transcription API** (Port 8000) - Dedicated FastAPI service for audio transcription with GPU pooling
3. **Frontend** (Port 80) - Web interface served via Nginx

All services are managed by **Supervisor** within the container.

## Directory Structure

```
KCC_app/
├── backend/                    # Main backend API
│   ├── app.py                 # FastAPI application
│   ├── config.py              # Configuration
│   ├── requirements.txt       # Python dependencies
│   ├── services/              # Service modules
│   ├── uploads/               # Uploaded audio files (mounted volume)
│   └── logs/                  # Application logs (mounted volume)
├── backend_transcription/     # Transcription API
│   ├── transcription_api.py   # FastAPI transcription service
│   ├── gpu_pool.py           # GPU pooling manager
│   ├── requirements.txt      # Python dependencies
│   └── uploads/              # Uploaded files (mounted volume)
├── frontend/                  # Web interface
│   ├── index.html
│   ├── script.js
│   ├── styles.css
│   └── config.js
├── notebook/                  # Data files
│   └── Cleaned_QA_Audio.xlsx # Q&A dataset (mounted volume)
├── logs/supervisor/           # Supervisor logs (mounted volume)
├── Dockerfile                 # Multi-service container definition
├── docker-compose.yml         # Docker Compose configuration
├── supervisord.conf          # Supervisor configuration
├── nginx.conf                # Nginx configuration
├── .env                      # Environment variables (create from .env.example)
└── README.md                 # This file
```

## Prerequisites

- Docker Engine (20.10+)
- Docker Compose (1.29+)
- NVIDIA Docker runtime (for GPU support)
- At least 16GB RAM
- NVIDIA GPU with CUDA support

## Setup

### 1. Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and set your values:

```env
HF_TOKEN=your_huggingface_token_here
TRANSLATION_API_BASE_URL=https://your-translation-api.ngrok-free.app
```

### 2. Build and Run

Build and start all services:

```bash
docker-compose up --build
```

Or run in detached mode:

```bash
docker-compose up -d --build
```

### 3. Access the Application

- **Frontend**: http://localhost
- **Backend API**: http://localhost:5000
- **Backend API Docs**: http://localhost:5000/docs
- **Transcription API**: http://localhost:8000
- **Transcription API Docs**: http://localhost:8000/docs

## API Endpoints

### Backend API (Port 5000)

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio file
- `POST /process-audio` - Complete workflow (transcribe + search)
- `POST /search` - Semantic search in Q&A dataset
- `GET /dataset-info` - Get Q&A dataset information
- `WS /ws` - WebSocket for live audio processing

### Transcription API (Port 8000)

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio using GPU pool
- `GET /status` - Get GPU pool status
- `GET /results/{request_id}` - Get transcription result by ID

## Docker Commands

### Start services
```bash
docker-compose up -d
```

### Stop services
```bash
docker-compose down
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f voice_recommendation_system

# Supervisor logs (inside container)
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/supervisord.log
```

### Restart services
```bash
docker-compose restart
```

### Rebuild after code changes
```bash
docker-compose up -d --build
```

### Access container shell
```bash
docker-compose exec voice_recommendation_system bash
```

## Volumes

The following directories are mounted as volumes for persistence:

- `./backend/uploads` - Uploaded audio files from main backend
- `./backend/logs` - Backend application logs
- `./backend_transcription/uploads` - Transcription API uploads
- `./notebook` - Q&A dataset and embeddings
- `./frontend` - Frontend files (read-only)
- `./logs/supervisor` - Supervisor service logs

## Monitoring

### Check Service Status

```bash
# Inside container
docker-compose exec voice_recommendation_system supervisorctl status

# Expected output:
# backend                          RUNNING   pid 123, uptime 0:01:00
# nginx                            RUNNING   pid 124, uptime 0:01:00
# transcription_api               RUNNING   pid 125, uptime 0:01:00
```

### Health Checks

The container has built-in health checks. Check status:

```bash
docker-compose ps
```

### View Individual Service Logs

```bash
# Backend logs
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/backend.out.log

# Transcription API logs
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/transcription_api.out.log

# Nginx logs
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/nginx.out.log
```

## Troubleshooting

### Services won't start

1. Check logs:
   ```bash
   docker-compose logs
   ```

2. Check supervisor status:
   ```bash
   docker-compose exec voice_recommendation_system supervisorctl status
   ```

3. Manually restart a service:
   ```bash
   docker-compose exec voice_recommendation_system supervisorctl restart backend
   ```

### GPU not detected

1. Verify NVIDIA Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check GPU in container:
   ```bash
   docker-compose exec voice_recommendation_system nvidia-smi
   ```

### Port conflicts

If ports 80, 5000, or 8000 are already in use, edit `docker-compose.yml`:

```yaml
ports:
  - "8080:80"      # Change frontend to port 8080
  - "5001:5000"    # Change backend to port 5001
  - "8001:8000"    # Change transcription to port 8001
```

### Out of memory

Increase Docker memory limits in Docker settings or add to `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

## Development

### Making Changes

1. **Frontend changes**: Edit files in `./frontend/` - they're mounted as a volume
2. **Backend changes**: Edit files in `./backend/` or `./backend_transcription/`, then rebuild:
   ```bash
   docker-compose up -d --build
   ```

### Testing

Test the API endpoints:

```bash
# Health check
curl http://localhost:5000/health

# Transcription API status
curl http://localhost:8000/status

# Upload and process audio
curl -X POST -F "audio=@test_audio.mp3" http://localhost:5000/process-audio
```

## Production Deployment

For production deployment:

1. Set strong passwords and tokens in `.env`
2. Use a reverse proxy (nginx/traefik) with SSL
3. Set up proper logging and monitoring
4. Configure firewall rules
5. Use Docker secrets for sensitive data
6. Enable resource limits in docker-compose.yml

## License

[Your License Here]

## Support

For issues and questions, please contact the development team.

