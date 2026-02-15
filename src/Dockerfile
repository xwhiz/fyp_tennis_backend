FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless


# Copy the application code
COPY . .

# Expose the port
EXPOSE 7000

# Use uvicorn to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]