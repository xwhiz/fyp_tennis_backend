FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 7000

# Use uvicorn to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]