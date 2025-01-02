# syntax = docker/dockerfile:experimental
FROM python:3.9

# Expose ports
EXPOSE 443
EXPOSE 80

# 1. Install Debian key packages (so that apt-get update can verify repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        debian-archive-keyring \
        gnupg2 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Install certbot/nginx dependencies
RUN apt-get update && \
    apt-get install -y python3-certbot-nginx && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Python requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy scripts
COPY start_server.sh /usr/bin/start_server

# 5. Copy context and install package
WORKDIR /root
COPY . .
RUN pip install --no-cache-dir .

# 6. Create models directory and environment variable
RUN mkdir /models
ENV MODELS_DIR /models

# 7. Set the default command
CMD ["start_server"]
