FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic telemetry data
# Load telemetry into the database
# Train the fraud/anomaly detection model
RUN python generate_data.py \
    && python scripts/load_data.py \
    && python -m scripts.train_model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
