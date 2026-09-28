FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# El contenedor no decide qué proceso correr: docker-compose.yml pasa el
# comando (api / dashboard / scheduler) para que las tres piezas compartan
# la misma imagen sin triplicar el Dockerfile.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
