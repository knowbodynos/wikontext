FROM python:3.9-bullseye

# Expose ports
EXPOSE 443
EXPOSE 80

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-certbot-nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY start_server.sh /usr/bin/start_server

# Copy context and install package
WORKDIR /root
COPY . .
RUN pip install --no-cache-dir .

# Create models directory and environment variable
RUN mkdir /models
ENV MODELS_DIR /models

CMD ["start_server"]