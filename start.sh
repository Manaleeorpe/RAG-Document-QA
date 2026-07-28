#!/bin/bash
set -e

# Start FastAPI backend in the background on port 8000
uvicorn app.main:app --host 0.0.0.0 --port "${FASTAPI_PORT:-8000}" &

# Start Streamlit on the port Render provides (defaults to 8501)
exec streamlit run streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
