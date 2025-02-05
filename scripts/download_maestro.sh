#!/bin/bash

# Download dataset
mkdir -p ./datasets/maestro
wget -O ./datasets/maestro/maestro-v3.0.0.zip https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip

# Unzip dataset
unzip ./downloaded_datasets/maestro/maestro-v3.0.0.zip -d ./datasets/
