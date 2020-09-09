#!/bin/bash

cd /app
mkdir models
aws s3 cp s3://wikontext/models models/ --recursive

cd /wikontext/flask
gunicorn --bind 0.0.0.0:5000 wikontext:app
