#!/bin/bash

if [ "$#" -ge 1 ]; then
  MODELS_DIR=$1
else
  MODELS_DIR=$(pwd)
fi

aws s3 cp s3://wikontext/models/wiki2vec.tar.gz ${MODELS_DIR}/wiki2vec.tar.gz
tar xzfv ${MODELS_DIR}/wiki2vec.tar.gz
