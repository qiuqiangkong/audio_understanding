# Music Generation/TTS with Large Language Models

This repository contains a tutorial of building audio understanding systems with large language models (LLMs). The audio understanding tasks include automatic speech recogntion (ASR), audio caption, audio query answering, music transcription, etc. The repository is written in PyTorch. All tasks are formatted to a same format with tuples of audio, question, and answering as input. An audio understanding system consists of an audio encoder and an LLM decoder. When loading pretrained audio encoders and train LLM decoders from scratch, users can train an audio understanding system in less than 10 hours using a single RTX 4090 GPU.

<img src="./assets/llm.png" width="600">

## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/audio_understanding
cd audio_understanding

# Install Python environment
conda create --name music_llm python=3.10

# Activate environment
conda activate audio_understanding

# Install Python packages dependencies
bash env.sh
```

## 0. Download datasets

To train the ASR system, download LibriSpeech dataset (1,000 hours)

```bash
bash ./scripts/download_librispeech.sh
```

## 1. Train

Train an ASR model:

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/asr_librispeech.yaml"
```

Train an audio caption model:
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/audio_caption_clotho.yaml"
```

Train a piano transcription model:
```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/piano_transcription_maestro.yaml"
```

Train music generation model:

```python
CUDA_VISIBLE_DEVICES=1 python train.py --config="./configs/gtzan.yaml"
```

The training takes around 10 hours to train for 100,000 steps on a single RTX4090 GPU.

![Training & Validation Loss](assets/result_loss.png)

### Train on Multiple GPUs.

We use Huggingface accelerate library to train the systems on multiple GPUs. train_accelerate.py just adds a few lines to train.py. Here is an example to run with 4 GPUs:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/ljspeech.yaml"
```

Then, the training can speed up by 4x times. The code can also train with multiple nodes such as 32 GPUs with 4 nodes.

## 2. Sample

Users can sample audio from text prompts using trained checkpoints:

```python
CUDA_VISIBLE_DEVICES=0 python sample.py \
	--config="./configs/ljspeech.yaml" \
	--ckpt_path="./checkpoints/train/ljspeech/step=100000.pth"
```

After training on 1 RTX4090 GPU for 100,000 stesp in 10 hours, the sampled audio sounds like:


| Task       | Training Dataset      | Text prompt                                                                                                   | Sample 1                                                                                      | Sample 2                                                                                                                                                                                                                                                                                                  |
|------------|-----------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| TTS        | LJSpeech (size: 24 h) | A happy dog ran through the park, wagging its tail excitedly, greeting everyone with joy and boundless energy | <video src="https://github.com/user-attachments/assets/5d7421e9-9f64-48a1-92c5-6cfee04a6e8c"> | <video src="https://github.com/user-attachments/assets/3433b3b7-2b48-42a9-a138-3b8166591a85"> |
| Music Gen  | GTZAN (size: 8 h)     | country                                                                                                       | <video src="https://github.com/user-attachments/assets/428dd426-787a-487b-9c32-197d61bfece3"> | <video src="https://github.com/user-attachments/assets/2655f774-7133-4a68-b3fc-11fd9786c79f"> |


























## External links

The Llama model code is from: https://github.com/qiuqiangkong/mini_llm

## License

MIT
