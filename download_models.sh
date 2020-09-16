#!/bin/bash

ROOT_DIR=$(dirname ${BASH_SOURCE[0]})
MODELS_DIR=${ROOT_DIR}/models

mkdir -p ${MODELS_DIR}
aws s3 cp s3://wikontext/models/wiki2vec.tar.gz ${MODELS_DIR}/wiki2vec.tar.gz
tar xzfv ${MODELS_DIR}/wiki2vec.tar.gz
