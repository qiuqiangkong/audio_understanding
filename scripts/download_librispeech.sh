#!/bin/bash

# Download dataset
mkdir -p ./downloaded_datasets/LibriSpeech
wget -O ./downloaded_datasets/LibriSpeech/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/dev-other.tar.gz https://www.openslr.org/resources/12/dev-other.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/test-other.tar.gz https://www.openslr.org/resources/12/test-other.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/train-clean-360.tar.gz https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -O ./downloaded_datasets/LibriSpeech/train-other-500.tar.gz https://www.openslr.org/resources/12/train-other-500.tar.gz

# Unzip dataset
mkdir -p ./datasets
tar -zxvf ./downloaded_datasets/LibriSpeech/dev-clean.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/dev-other.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/test-clean.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/test-other.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/train-clean-100.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/train-clean-360.tar.gz -C ./datasets/
tar -zxvf ./downloaded_datasets/LibriSpeech/train-other-500.tar.gz -C ./datasets/
