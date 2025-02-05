#!/bin/bash

# Download dataset
mkdir -p ./datasets/librispeech
wget -O ./datasets/librispeech/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
wget -O ./datasets/librispeech/dev-other.tar.gz https://www.openslr.org/resources/12/dev-other.tar.gz
wget -O ./datasets/librispeech/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
wget -O ./datasets/librispeech/test-other.tar.gz https://www.openslr.org/resources/12/test-other.tar.gz
wget -O ./datasets/librispeech/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -O ./datasets/librispeech/train-clean-360.tar.gz https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -O ./datasets/librispeech/train-other-500.tar.gz https://www.openslr.org/resources/12/train-other-500.tar.gz

# Unzip dataset
tar -zxvf ./datasets/librispeech/dev-clean.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/dev-other.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/test-clean.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/test-other.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/train-clean-100.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/train-clean-360.tar.gz -C ./datasets/
tar -zxvf ./datasets/librispeech/train-other-500.tar.gz -C ./datasets/
