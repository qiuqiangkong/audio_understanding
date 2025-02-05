#!/bin/bash

# Download dataset
mkdir -p ./datasets/gtzan
wget -O ./datasets/gtzan/genres.tar.gz https://huggingface.co/datasets/qiuqiangkong/gtzan/resolve/main/genres.tar.gz?download=true

# Unzip dataset
tar -zxvf ./datasets/gtzan/genres.tar.gz -C ./datasets/gtzan/