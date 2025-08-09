FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of your application's code into the container.
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main5:app", "--host", "0.0.0.0", "--port", "8080"]
