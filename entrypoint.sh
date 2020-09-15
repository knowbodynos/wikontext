#!/bin/bash

if ! [ -d ${MODELS_DIR}/wiki2vec ]; then
  if ! [ -f ${MODELS_DIR}/wiki2vec.tar.gz ]; then
    aws s3 cp s3://wikontext/models/wiki2vec.tar.gz ${MODELS_DIR}/wiki2vec.tar.gz
  else
    tar xzfv ${MODELS_DIR}/wiki2vec.tar.gz
  fi
fi

gunicorn --bind 0.0.0.0:80 -t 180 -w 1 wikontext.wsgi:app
