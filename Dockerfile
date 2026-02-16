FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get install -y netcat

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*



RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN pip install --upgrade pip
RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

COPY . .

EXPOSE 7000
