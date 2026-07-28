FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

# Streamlit (public-facing) on 8501, FastAPI (internal) on 8000
EXPOSE 8501

CMD ["./start.sh"]
