#!/bin/bash

# Download dataset
mkdir -p ./downloaded_datasets/maestro
wget -O ./downloaded_datasets/maestro/maestro-v3.0.0.zip https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

# Unzip dataset
mkdir -p ./datasets
unzip ./downloaded_datasets/maestro/maestro-v3.0.0.zip -d ./datasets/
