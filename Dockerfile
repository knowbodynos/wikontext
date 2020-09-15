# syntax = docker/dockerfile:experimental
FROM python:3

EXPOSE 443
EXPOSE 80

RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /models
ENV MODELS_DIR /models

WORKDIR /root
COPY . .

RUN mv wikontext.nginx /etc/nginx/sites-enabled/wikontext.nginx

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir .

CMD ["./entrypoint.sh"]
