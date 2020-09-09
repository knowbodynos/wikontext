# syntax = docker/dockerfile:experimental
FROM python:3

EXPOSE 443
EXPOSE 80

RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt

COPY wikontext /wikontext
COPY wikontext.nginx /etc/nginx/sites-enabled/wikontext.nginx

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
