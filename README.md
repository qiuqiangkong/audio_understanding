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

## 1. Train & Evaluate

### 1.1 Music tagging

To train a music tagging system, users need to do download the GTZAN dataset (1.3 GB, 8 hours)

```bash
bash ./scripts/download_gtzan.sh
```

```python
# Train (Takes 15 min on 1 RTX4090 to train for 10,000 steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/music_tagging_gtzan.yaml"

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/music_tagging_gtzan.yaml" \
	--ckpt_path="./checkpoints/train/music_tagging_gtzan/step=20000.pth" \
	--audio_path="./assets/gtzan_blues.00002.au"
```

### 1.2 Automatic speech recognition (ASR)

To train an ASR system, users need to do download the LibriSpeech dataset (1,000 hours)

```bash
bash ./scripts/download_librispeech.sh
```

```python
# Train (Takes ~8 hours on 1 RTX4090 to train for 100,000 steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/asr_librispeech.yaml"

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/asr_librispeech.yaml" \
	--ckpt_path="./checkpoints/train/asr_librispeech/step=20000.pth" \
	--audio_path="./assets/librispeech_1688-142285-0000.flac"
```

### 1.3 Audio Caption
```bash
bash ./scripts/download_clotho.sh
```

```python
# Train (takes ~8 hours on 1 RTX4090 to train for 100,000 steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/audio_caption_clotho.yaml"

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/audio_caption_clotho.yaml" \
	--ckpt_path="./checkpoints/train/audio_caption_clotho/step=20000.pth" \
	--audio_path="./assets/clotho_birds_long.wav"
```

### 1.4 Piano Transcription
```bash
bash ./scripts/download_maestro.sh
```

```python
# Train (takes ~8 hours on 1 RTX4090 to train for 100,000 steps)
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/piano_transcription_maestro.yaml"

# Inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--config="./configs/piano_transcription_maestro.yaml" \
	--ckpt_path="./checkpoints/train/piano_transcription_maestro/step=20000.pth" \
	--audio_path="./assets/clotho_birds_long.wav"
```

![Training & Validation Loss](assets/result_loss.png)

## 2. Train on Multiple GPUs.

We use Huggingface accelerate library to train the systems on multiple GPUs. train_accelerate.py just adds a few lines to train.py. Here is an example to run with 4 GPUs:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_accelerate.py --config="./configs/asr_librispeech.yaml"
```

Then, the training can speed up by 4x times. The code can also train with multiple nodes such as 32 GPUs with 4 nodes.

After training on 1 RTX4090 GPU for 100,000 stesp in 10 hours, the sampled audio sounds like:


| Task                | Training Dataset            | Input audio                                                              | Output                                                                                                                                               | Ground truth                                                                                                                                       |
|---------------------|-----------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------|
| Music Tagging       | GTZAN (size: 8 h)           |                                                                          | blues                                                                                                                                                | blues                                                                                                                                              |
| ASR                 | LibriSpeech (size: 1,000 h) |                                                                          | there ' s iron they say in all our blood and a grain or two perhaps is good but his he makes me harshly feel has got a little too much of steel anon | THERE'S IRON THEY SAY IN ALL OUR BLOOD AND A GRAIN OR TWO PERHAPS IS GOOD BUT HIS HE MAKES ME HARSHLY FEEL HAS GOT A LITTLE TOO MUCH OF STEEL ANON |
| Audio Caption       | Clotho (size: 24 h)         |                                                                          | a variety of birds are chirping while the birds are chirping in the background and the birds are chirping loudly.                                    | bird is chirping continuously as time goes on.                                                                                                     |
| Piano Transcription | MAESTRO (199 h)             |                                                                          | 

























## External links

The Llama model code is from: https://github.com/qiuqiangkong/mini_llm

## License

MIT
