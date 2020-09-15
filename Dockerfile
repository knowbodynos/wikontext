# syntax = docker/dockerfile:experimental
FROM python:3

# Expose ports
EXPOSE 443
EXPOSE 80

# Install dependencies
RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Configure nginx
COPY wikontext.nginx /etc/nginx/sites-enabled/wikontext.nginx

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy context and install package
WORKDIR /root
COPY . .
RUN pip install --no-cache-dir .

# Create models directory and environment variable
RUN mkdir /models
ENV MODELS_DIR /models

CMD ["./entrypoint.sh"]
