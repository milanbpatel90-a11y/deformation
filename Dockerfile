FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY assets ./assets
COPY models ./models
COPY run.py ./run.py
RUN mkdir -p /app/runtime /app/output
EXPOSE 8000
CMD ["uvicorn","backend.api.main:app","--host","0.0.0.0","--port","8000"]
