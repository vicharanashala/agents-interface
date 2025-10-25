# Docker Volumes Configuration

## 📦 Mounted Volumes

This Docker Compose setup uses **bind mounts** to persist data outside the container and enable easy development.

## Volume Mappings

| Host Path | Container Path | Purpose | Read/Write |
|-----------|---------------|---------|------------|
| `./backend/uploads` | `/app/backend/uploads` | Backend uploaded audio files | RW |
| `./backend/logs` | `/app/backend/logs` | Backend application logs | RW |
| `./backend_transcription/uploads` | `/app/backend_transcription/uploads` | Transcription API uploads | RW |
| `./notebook` | `/app/notebook` | Q&A dataset (Excel file) | RW |
| `./frontend` | `/app/frontend` | Frontend HTML/CSS/JS files | RO |
| `./logs/supervisor` | `/var/log/supervisor` | Supervisor service logs | RW |

## Volume Details

### 1. Backend Uploads (`./backend/uploads`)

**Purpose**: Stores audio files uploaded to the main backend API

**Contents**:
- Temporary audio files during processing
- Files are typically deleted after processing

**Size**: Varies based on usage, typically < 1GB

### 2. Backend Logs (`./backend/logs`)

**Purpose**: Application logs from the main backend

**Contents**:
- `app_performance.log` - Performance metrics
- `app_startup.log` - Startup logs
- `backend.log` - General application logs

**Size**: Can grow over time, monitor and rotate

### 3. Transcription Uploads (`./backend_transcription/uploads`)

**Purpose**: Temporary storage for transcription API

**Contents**:
- Audio files being transcribed
- Automatically cleaned after processing

**Size**: Varies, typically < 1GB

### 4. Notebook Data (`./notebook`)

**Purpose**: Q&A dataset and embeddings

**Contents**:
- `Cleaned_QA_Audio.xlsx` - Main Q&A dataset
- Generated embeddings (if any)
- FAISS indices (if generated)

**Size**: ~10-100MB depending on dataset

**Important**: This is the core data for semantic search!

### 5. Frontend Files (`./frontend` - Read Only)

**Purpose**: Web interface files

**Contents**:
- `index.html` - Main page
- `script.js` - Frontend logic
- `styles.css` - Styling
- `config.js` - Configuration

**Read-only**: Prevents accidental modification from container

**Development**: Edit on host, changes reflect immediately

### 6. Supervisor Logs (`./logs/supervisor`)

**Purpose**: Service management logs

**Contents**:
- `supervisord.log` - Main supervisor log
- `backend.out.log` - Backend stdout
- `backend.err.log` - Backend stderr
- `transcription_api.out.log` - Transcription stdout
- `transcription_api.err.log` - Transcription stderr
- `nginx.out.log` - Nginx stdout
- `nginx.err.log` - Nginx stderr

**Size**: Monitor and rotate periodically

## Benefits of Volume Mounting

### ✅ Data Persistence
- Data survives container restarts
- Data survives container deletion
- Easy backup (just copy the directories)

### ✅ Easy Development
- Edit code on host
- Changes reflect in container (for frontend)
- No need to rebuild for data changes

### ✅ Easy Debugging
- View logs directly on host
- Inspect uploads on host
- No need to enter container

### ✅ Easy Backup
- Simply backup the mounted directories
- Use standard backup tools
- Version control friendly

## Managing Volumes

### Backup All Data

```bash
cd /home/aic_u2/Shubhankar
tar -czf KCC_app_backup_$(date +%Y%m%d).tar.gz \
  KCC_app/backend/uploads \
  KCC_app/backend/logs \
  KCC_app/backend_transcription/uploads \
  KCC_app/notebook \
  KCC_app/logs
```

### Clean Up Uploads (Free Space)

```bash
# Clean backend uploads (older than 7 days)
find ./backend/uploads -type f -mtime +7 -delete

# Clean transcription uploads (older than 7 days)
find ./backend_transcription/uploads -type f -mtime +7 -delete
```

### Rotate Logs

```bash
# Archive old logs
cd logs/supervisor
tar -czf logs_archive_$(date +%Y%m%d).tar.gz *.log
rm *.log
```

### Reset Everything (Fresh Start)

```bash
# Stop container
docker-compose down

# Remove all data (CAUTION: This deletes everything!)
rm -rf backend/uploads/* backend/logs/* backend_transcription/uploads/* logs/supervisor/*

# Restart
docker-compose up -d
```

## Disk Space Monitoring

### Check Volume Sizes

```bash
# Overall size
du -sh backend/uploads backend/logs backend_transcription/uploads notebook logs

# Detailed breakdown
du -h --max-depth=1 .
```

### Auto-cleanup (Optional)

Create a cron job to automatically clean old uploads:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 2 AM)
0 2 * * * find /home/aic_u2/Shubhankar/KCC_app/backend/uploads -type f -mtime +7 -delete
```

## Security Considerations

### File Permissions

The volumes maintain host file permissions. Ensure proper permissions:

```bash
# Set proper ownership
chown -R $USER:$USER /home/aic_u2/Shubhankar/KCC_app

# Set directory permissions
chmod 755 backend/uploads backend_transcription/uploads notebook
chmod 644 notebook/Cleaned_QA_Audio.xlsx
```

### Sensitive Data

- **Never commit** uploads or logs to version control
- The `.dockerignore` file excludes these directories
- Add to `.gitignore` if using git:
  ```
  backend/uploads/
  backend/logs/
  backend_transcription/uploads/
  logs/
  ```

## Troubleshooting Volumes

### Permission Denied Errors

```bash
# Fix ownership
sudo chown -R $(id -u):$(id -g) /home/aic_u2/Shubhankar/KCC_app
```

### Volume Not Mounting

```bash
# Check if paths exist
ls -la backend/uploads backend/logs

# Recreate if needed
mkdir -p backend/uploads backend/logs backend_transcription/uploads logs/supervisor

# Restart container
docker-compose restart
```

### Container Can't Write to Volume

```bash
# Check permissions
ls -ld backend/uploads

# Should show: drwxrwxr-x

# Fix if needed
chmod 775 backend/uploads
```

## Best Practices

1. **Regular Backups**: Backup `notebook/` directory regularly (contains core data)
2. **Log Rotation**: Set up log rotation to prevent disk full
3. **Monitor Disk Usage**: Check disk space regularly
4. **Clean Old Uploads**: Delete old temporary files
5. **Version Control**: Keep `notebook/Cleaned_QA_Audio.xlsx` in version control or backup

## Summary

The volume configuration provides:
- ✅ **Persistence** - Data survives restarts
- ✅ **Performance** - Direct host filesystem access
- ✅ **Flexibility** - Easy to edit and manage
- ✅ **Portability** - Easy to backup and restore
- ✅ **Development** - Live changes for frontend

All volumes are created automatically when you run `docker-compose up`.

