#!/bin/bash

DOCKER_BUILDKIT=1 docker build -t wikontext .
docker run -it -d -p 80:80 -p 443:443 \
    -e CERT_EMAIL=${CERT_EMAIL} \
    -v $(pwd)/letsencrypt:/etc/letsencrypt \
    -v $(pwd)/nginx/sites-available:/etc/nginx/sites-available \
    -v $(pwd)/nginx/sites-enabled:/etc/nginx/sites-enabled \
    -v $(pwd)/models:/models \
    wikontext

