#!/bin/bash

gunicorn --bind 0.0.0.0:80 --bind 0.0.0.0:443 --timeout 180 wikontext.wsgi:app
