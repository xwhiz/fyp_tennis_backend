FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip


RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN pip install --upgrade pip
RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY . .

EXPOSE 7000
