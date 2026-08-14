FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate telemetry, load it into the database,
# then train the fraud/anomaly model.
RUN python generate_data.py \
    && python -m scripts.load_data \
    && python -m scripts.train_model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
