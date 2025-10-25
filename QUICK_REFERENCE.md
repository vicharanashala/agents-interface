# Quick Reference Card

## 🚀 Start/Stop

```bash
# Start
./start.sh
# or
docker-compose up -d

# Stop
./stop.sh
# or
docker-compose down
```

## 🌐 URLs

- Frontend: http://localhost
- Backend API: http://localhost:5000/docs
- Transcription API: http://localhost:8000/docs

## 📊 Status Check

```bash
# Container status
docker-compose ps

# Service status (inside container)
docker-compose exec voice_recommendation_system supervisorctl status

# Health checks
curl http://localhost:5000/health
curl http://localhost:8000/health
```

## 📝 View Logs

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f voice_recommendation_system

# Backend log
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/backend.out.log

# Transcription log
docker-compose exec voice_recommendation_system tail -f /var/log/supervisor/transcription_api.out.log
```

## 🔄 Restart

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose exec voice_recommendation_system supervisorctl restart backend
docker-compose exec voice_recommendation_system supervisorctl restart transcription_api
docker-compose exec voice_recommendation_system supervisorctl restart nginx
```

## 🛠️ Debug

```bash
# Enter container
docker-compose exec voice_recommendation_system bash

# Check GPU
docker-compose exec voice_recommendation_system nvidia-smi

# Check Python
docker-compose exec voice_recommendation_system python --version
```

## 🧪 Test API

```bash
# Backend health
curl http://localhost:5000/health

# Process audio
curl -X POST -F "audio=@test.mp3" http://localhost:5000/process-audio

# Transcription status
curl http://localhost:8000/status
```

## 📦 Rebuild

```bash
# After code changes
docker-compose up -d --build
```

## 🗑️ Clean Up

```bash
# Stop and remove containers
docker-compose down

# Stop, remove containers and volumes
docker-compose down -v

# Remove all (including images)
docker-compose down -v --rmi all
```

