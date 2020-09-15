#!/bin/bash

./download_models.sh

gunicorn --bind 0.0.0.0:80 -t 180 -w 1 wikontext.wsgi:app
