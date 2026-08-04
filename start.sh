#!/bin/bash
set -e

export PYTHONPATH=/app/app:$PYTHONPATH

FASTAPI_PORT_NUM=${FASTAPI_PORT:-8000}

# Start FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port "$FASTAPI_PORT_NUM" &
FASTAPI_PID=$!

# Wait up to 120 s for FastAPI to bind to its port
echo "Waiting for FastAPI on port $FASTAPI_PORT_NUM..."
for i in $(seq 1 60); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost', $FASTAPI_PORT_NUM)); s.close()" 2>/dev/null; then
    echo "FastAPI is ready."
    break
  fi
  if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    echo "ERROR: FastAPI process exited unexpectedly. Check your OPENAI_API_KEY, DEEPSEEK_API_KEY, and PINECONE_API_KEY."
    exit 1
  fi
  sleep 2
done

# Start Streamlit on the port Railway provides (defaults to 8501)
exec streamlit run streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
