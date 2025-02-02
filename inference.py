"""Modified from https://github.com/qiuqiangkong/mini_llm/blob/main/sample.py
"""
from __future__ import annotations
import argparse

import pandas as pd
from pathlib import Path
import soundfile
import torch
import librosa
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop, StartCrop
from audidata.transforms import Mono
from audidata.datasets import MAESTRO

from audio_understanding.utils import parse_yaml
from train import get_audio_encoder, get_tokenizer, get_llm


def inference(args):

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path
    audio_path = args.audio_path
    device = "cuda"

    # Default parameters
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    clip_samples = round(clip_duration * sr)
    temperature = 1.0
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path)

    # Audio encoder
    audio_encoder = get_audio_encoder(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Tokenizer for converting text into IDs and vice versa
    tokenizer = get_tokenizer(configs=configs)
    
    # LLM decoder
    llm = get_llm(
        configs=configs, 
        audio_latent_dim=audio_encoder.latent_dim, 
        vocab_size=len(tokenizer),
        ckpt_path=ckpt_path
    ).to(device)

    # Load the begining part of audio
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = audio[0 : clip_samples]
    audio = librosa.util.fix_length(data=audio, size=clip_samples, axis=0)
    audio = torch.Tensor(audio[None, None, :]).to(device)  # shape: (b, c, t)

    # Encode audio into latent
    audio_latent = audio_encoder.encode(audio=audio)  # shape: (b, t, d)

    question = get_question(config_yaml)  # str
    batch_question = [question]  # shape: (b,)

    # Tokenize question text to IDs
    question_ids = tokenizer.texts_to_ids(
        texts=batch_question, 
        fix_length=configs["max_question_len"]
    ).to(device)  # shape: (b, t)

    # Tokenize answering text to IDs
    start_token_id = tokenizer.cls_token_id  # 101
    answering_ids = torch.LongTensor([[start_token_id]]).to(device)  # (b, 1)

    # Prepare inputs
    seqs = [audio_latent, question_ids, answering_ids]
    seq_types = ["audio", "id", "id"]

    with torch.no_grad():
        llm.eval()
        output_seqs = llm.generate(
            seqs=seqs,
            seq_types=seq_types,
            max_new_ids=configs["max_answering_len"], 
            # max_new_ids=300,
            temperature=temperature, 
            top_k=get_top_k(config_yaml)
        )

    answering_ids = output_seqs[2][0]  # shape: (t)

    answering_texts = convert_ids_to_texts(config_yaml, tokenizer, answering_ids)
    print("Output: {}".format(answering_texts))


def get_top_k(config_yaml: str) -> int:

    if Path(config_yaml).stem in ["asr_librispeech"]:
        return 1

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        return 20

    elif Path(config_yaml).stem in ["audio_caption_clotho"]:
        return 100

    else:
        print("Using default top_k=100")
        return 100

def get_question(config_yaml: str) -> str:
    
    if Path(config_yaml).stem in ["asr_librispeech"]:
        return "Automatic speech recognition."

    elif Path(config_yaml).stem in ["audio_caption_clotho"]:
        return "Audio caption."

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        return "Music transcription."

    else:
        raise NotImplementedError("Users need to write a question!")


def convert_ids_to_texts(config_yaml, tokenizer, answering_ids):

    if Path(config_yaml).stem in ["asr_librispeech", "audio_caption_clotho"]:
        return tokenizer.tok.decode(answering_ids, skip_special_tokens=True)

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        a1 = tokenizer.tok.convert_ids_to_tokens(answering_ids)
        import pickle
        pickle.dump(a1, open("_zz.pkl", "wb"))
        from IPython import embed; embed(using=False); os._exit(0)

    else:
        print("Using default ids to texts conversion.")
        return tokenizer.tok.decode(answering_ids, skip_special_tokens=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)