#!/bin/bash

# Download dataset
mkdir -p ./datasets/clotho

wget -O ./datasets/clotho/clotho_audio_development.7z https://zenodo.org/records/3490684/files/clotho_audio_development.7z?download=1
wget -O ./datasets/clotho/clotho_audio_evaluation.7z https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z?download=1
wget -O ./datasets/clotho/clotho_captions_development.csv https://zenodo.org/records/3490684/files/clotho_captions_development.csv?download=1
wget -O ./datasets/clotho/clotho_captions_evaluation.csv https://zenodo.org/records/3490684/files/clotho_captions_evaluation.csv?download=1
wget -O ./datasets/clotho/clotho_metadata_development.csv https://zenodo.org/records/3490684/files/clotho_metadata_development.csv?download=1
wget -O ./datasets/clotho/clotho_metadata_evaluation.csv https://zenodo.org/records/3490684/files/clotho_metadata_evaluation.csv?download=1
wget -O ./datasets/clotho/LICENSE https://zenodo.org/records/3490684/files/LICENSE?download=1

# Unzip dataset
7z e ./datasets/clotho/clotho_audio_development.7z -o./datasets/clotho/clotho_audio_development
7z e ./datasets/clotho/clotho_audio_evaluation.7z -o./datasets/clotho/clotho_audio_evaluation

