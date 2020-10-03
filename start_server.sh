#!/bin/bash

if [ -z ${CERT_EMAIL} ]; then
    EMAIL_FLAG="--register-unsafely-without-email"
else
    EMAIL_FLAG="-m ${CERT_EMAIL}"
fi

cp /etc/nginx/sites-available/wikontext.us.conf /etc/nginx/sites-enabled/

service nginx start
certbot --nginx -n --agree-tos ${EMAIL_FLAG} -d wikontext.us -d www.wikontext.us
service nginx restart

gunicorn --bind 0.0.0.0:8000 --timeout 60 wikontext.wsgi:app
