# KCC Voice Recommendation System - Setup Summary

## ✅ What Was Created

A complete Dockerized voice recommendation system with all components running in a single container.

### 📁 Directory Structure

```
/home/aic_u2/Shubhankar/KCC_app/
├── backend/                      # Main FastAPI backend (port 5020)
│   ├── app.py                   # Main application
│   ├── config.py                # Configuration
│   ├── services/                # Service modules
│   └── requirements.txt         # Dependencies
├── backend_transcription/        # Transcription API (port 8020)
│   ├── transcription_api.py     # Transcription service
│   ├── gpu_pool.py             # GPU pooling
│   └── requirements.txt        # Dependencies
├── frontend/                     # Web UI (port 80)
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── notebook/                     # Data files
│   └── Cleaned_QA_Audio.xlsx   # Q&A dataset
├── logs/supervisor/              # Service logs
├── Dockerfile                    # Multi-service container
├── docker-compose.yml           # Compose configuration
├── supervisord.conf             # Process manager config
├── nginx.conf                   # Web server config
├── start.sh                     # Quick start script
├── stop.sh                      # Stop script
├── .env                         # Environment variables
└── README.md                    # Full documentation
```

## 🚀 Quick Start


### Using Docker Compose 

```bash
cd /home/aic_u2/Shubhankar/KCC_app
docker compose up -d --build
```

## 🌐 Access the Services

Once started, access the services at:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost | Web interface |
| **Backend API** | http://localhost:5020 | Main API |
| **Backend Docs** | http://localhost:5020/docs | API documentation |
| **Transcription API** | http://localhost:8020 | Transcription service |
| **Transcription Docs** | http://localhost:8020/docs | Transcription API docs |

## 📦 What's Running Inside the Container

The single container runs three services managed by **Supervisor**:

1. **Backend** - FastAPI app with voice processing, search, translation
2. **Transcription API** - GPU-accelerated audio transcription
3. **Nginx** - Serves frontend and proxies API requests

## 💾 Volumes (Data Persistence)

The following directories are mounted as volumes and persist outside the container:

- `./backend/uploads` - Uploaded audio files
- `./backend/logs` - Application logs  
- `./backend_transcription/uploads` - Transcription uploads
- `./notebook` - Q&A dataset
- `./logs/supervisor` - Supervisor logs

This means your data is **safe** even if you stop or remove the container.

## 🔍 Monitoring & Logs

### Check if services are running

```bash
docker compose ps
```

### View all logs

```bash
docker compose logs -f
```

### View specific service logs

```bash
# Backend
docker compose exec voice_recommendation_system tail -f /var/log/supervisor/backend.out.log

# Transcription API
docker compose exec voice_recommendation_system tail -f /var/log/supervisor/transcription_api.out.log

# Nginx
docker compose exec voice_recommendation_system tail -f /var/log/supervisor/nginx.out.log
```

### Check supervisor status

```bash
docker compose exec voice_recommendation_system supervisorctl status
```

Expected output:
```
backend                          RUNNING   pid 123, uptime 0:05:00
transcription_api               RUNNING   pid 124, uptime 0:05:00
nginx                            RUNNING   pid 125, uptime 0:05:00
```

## 🛠️ Common Commands

### Start services
```bash
./start.sh
# or
docker-compose up -d
```

### Stop services
```bash
./stop.sh
# or
docker-compose down
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

### View real-time logs
```bash
docker-compose logs -f
```

## 🧪 Testing the APIs

### Test Backend API

```bash
# Health check
curl http://localhost:5000/health

# Get dataset info
curl http://localhost:5000/dataset-info
```

### Test Transcription API

```bash
# Health check
curl http://localhost:8000/health

# Get status
curl http://localhost:8000/status
```

### Test with an audio file

```bash
curl -X POST -F "audio=@your_audio_file.mp3" http://localhost:5000/process-audio
```

## ⚙️ Configuration

### Environment Variables

Edit `.env` file to change:

```env
HF_TOKEN=your_hugging_face_token
TRANSLATION_API_BASE_URL=your_translation_api_url
TRANSCRIPTION_API_URL=http://localhost:8000
```

### Port Changes

To change ports, edit `docker-compose.yml`:

```yaml
ports:
  - "8080:80"      # Frontend
  - "5001:5000"    # Backend
  - "8001:8000"    # Transcription
```

## 🔧 Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Check supervisor
docker-compose exec voice_recommendation_system supervisorctl status

# Restart a specific service
docker-compose exec voice_recommendation_system supervisorctl restart backend
```

### GPU not detected

```bash
# Check if NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check GPU in your container
docker-compose exec voice_recommendation_system nvidia-smi
```

### Port already in use

Stop the conflicting service or change ports in `docker-compose.yml`.

### Models not loading

Check if you have enough disk space and memory. The models require several GB.

## 📊 Resource Requirements

- **RAM**: 16GB minimum (32GB recommended)
- **Disk**: 20GB minimum for models and data
- **GPU**: NVIDIA GPU with CUDA support (for transcription)
- **CPU**: 4+ cores recommended

## 🎯 Next Steps

1. **Customize the frontend** - Edit files in `./frontend/`
2. **Add your Q&A data** - Update `./notebook/Cleaned_QA_Audio.xlsx`
3. **Configure APIs** - Update `.env` with your API tokens
4. **Deploy to production** - Set up reverse proxy with SSL

## 📝 Important Notes

- **First startup** may take 5-10 minutes as models download and load
- **GPU memory** is managed by the GPU pool in the transcription service
- **Uploads** are automatically cleaned up after processing
- **Logs** are rotated to prevent disk space issues

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify all services are running: `docker-compose ps`
3. Check supervisor status: `docker-compose exec voice_recommendation_system supervisorctl status`
4. Review the full README.md for detailed information

## ✨ Features

- ✅ All services in one container
- ✅ Automatic GPU pooling for transcription
- ✅ Multi-language support (22+ Indian languages + English)
- ✅ Semantic search in Q&A dataset
- ✅ Real-time WebSocket support
- ✅ Automatic health checks
- ✅ Persistent data volumes
- ✅ Easy deployment with Docker Compose

---

**Created on:** $(date)
**Location:** /home/aic_u2/Shubhankar/KCC_app

