#!/bin/bash

cp /etc/nginx/sites-available/wikontext.us.conf /etc/nginx/sites-enabled/
certbot --nginx -n -d wikontext.us -d www.wikontext.us
service nginx restart
gunicorn --bind 0.0.0.0:8000 --timeout 60 wikontext.wsgi:app
