#!/bin/bash
set -e

echo "=== Voice Recommendation System Startup ==="

# Create HuggingFace cache directory structure if it doesn't exist
CACHE_DIR="/root/.cache/huggingface"
MODELS_DIR="$CACHE_DIR/hub/models--ai4bharat--Cadence/snapshots/defe16c64db71b6b4a43f0125d436b57aa806670"
MODULES_DIR="$CACHE_DIR/modules/transformers_modules/ai4bharat/Cadence/defe16c64db71b6b4a43f0125d436b57aa806670"

# Create directories
mkdir -p "$MODELS_DIR"
mkdir -p "$MODULES_DIR"

# Copy fixed punctuation model file if it exists
if [ -f "/app/backend/punctuation_model_fix_/modeling_gemma3_punctuation.py" ]; then
    echo "Copying fixed punctuation model configuration..."
    cp /app/backend/punctuation_model_fix_/modeling_gemma3_punctuation.py "$MODELS_DIR/" 2>/dev/null || true
    cp /app/backend/punctuation_model_fix_/modeling_gemma3_punctuation.py "$MODULES_DIR/" 2>/dev/null || true
    echo "✓ Punctuation model fix applied"
else
    echo "⚠ Punctuation model fix file not found (will be applied if model downloads)"
fi

# Ensure backend directories exist
mkdir -p /app/backend/uploads /app/backend/logs

# Verify embedding data is accessible
if [ -d "/app/embedding_data/all_crops" ]; then
    echo "✓ Embedding data found at /app/embedding_data/all_crops"
    ls -lh /app/embedding_data/all_crops/*.faiss 2>/dev/null || echo "⚠ FAISS index not found"
else
    echo "⚠ WARNING: Embedding data not found at /app/embedding_data/all_crops"
fi

echo "✓ Backend directories ready"

echo "=== Starting Supervisor ==="
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

