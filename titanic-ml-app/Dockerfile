FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model is already trained and included in model/titanic_model.pkl.
# Uncomment to retrain the model at build time instead:
# RUN python train.py

EXPOSE 8760

CMD ["python", "app.py"]
