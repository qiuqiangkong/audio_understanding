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

# from data.text_normalization import TextNormalization
# from data.text_tokenization import BertTokenizer
# from train import get_audio_encoder, get_llm_decoder, get_audio_latent

from audio_understanding.utils import parse_yaml
from train import get_audio_encoder, get_tokenizer, get_llm


def inference(args):

    # Arguments

    # Arguments and parameters
    config_yaml = args.config_yaml
    ckpt_path = args.ckpt_path

    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]

    # Default parameters
    device = "cuda"
    split = "test"
    max_length = 500  # Max caption length
    # clip_duration = 10.  # Audio clip duration
    clip_duration = configs["clip_duration"]
    clip_samples = round(clip_duration * sr)
    # audio_encoder_name = "Cnn14"
    # llm_decoder_name = "Llama"
    num_samples = 1
    temperature = 1.0
    # top_k = 200
    # top_k = 20 # ASR
    top_k = 1

    # Dataset
    root = "/datasets/clotho"

    # Audio Cropper
    # crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)

    # # Caption transforms
    # target_transform = [
    #     TextNormalization(),  # Remove punctuations
    #     BertTokenizer(max_length=max_length)  # Convert captions to token IDs
    # ]
    # tokenizer = target_transform[1].tokenizer
    # start_token_id = tokenizer.cls_token_id  # 101
    # text_vocab_size = tokenizer.vocab_size  # 30,522

    # # # Dataset
    # meta_dict = get_clotho_meta(root, split)
    # audios_num = len(meta_dict["audio_name"])

    # # Load audio encoder
    # audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    # audio_encoder.to(device)

    ckpt = torch.load(ckpt_path)

    root = "/datasets/maestro-v3.0.0"
    dataset = MAESTRO(
        root=root,
        split="test",
        sr=sr,
        crop=StartCrop(clip_duration=10.),
        transform=Mono(),
        load_target=False,
    )

    # Audio encoder: Used to convert audio into latent
    audio_encoder = get_audio_encoder(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # Tokenizer: Used to convert text or audio codes into IDs and vice versa
    tokenizer = get_tokenizer(
        configs=configs, 
        ckpt_path=ckpt_path
    ).to(device)
    
    # LLM decoder
    llm = get_llm(
        configs=configs, 
        audio_latent_dim=audio_encoder.latent_dim, 
        vocab_size=len(tokenizer),
        ckpt_path=ckpt_path
    ).to(device)

    llm.to(device)

    # Text start token
    start_token_id = tokenizer.cls_token_id  # 101
    caption_ids = torch.LongTensor([[start_token_id]]).to(device)  # (b, 1)

    #
    audio_paths = dataset.meta_dict["audio_path"]
    # audio_paths = dataset.meta_dict["audio_path"][100:]
    # audio_paths = 

    # audio_paths = ["/datasets/LJSpeech-1.1/wavs/LJ049-0022.wav"]

    # audio_paths = ["/datasets/clotho/clotho_audio_evaluation/01862 heavy machine working.wav"]
    # audio_paths = ["/datasets/clotho/clotho_audio_evaluation/Footsteps on snow.wav"]
    # audio_paths = ["/datasets/librispeech/test-other/1688/142285/1688-142285-0000.flac"] 

    for audio_path in audio_paths:

        audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
        audio = audio[0 : clip_samples]
        audio = librosa.util.fix_length(data=audio, size=clip_samples, axis=0)

        # Move data to device
        audio = torch.Tensor(audio[None, None, :]).to(device)  # shape: (b, c, t_audio)

        # Encode audio into discrete codes
        audio = audio.to(device)
        audio_latent = audio_encoder.encode(audio=audio)  # shape: (b, t_code, q)

        seqs = [audio_latent, caption_ids]
        seq_types = ["audio", "id"]

        with torch.no_grad():
            llm.eval()

            output_seqs = llm.generate(
                seqs=seqs,
                seq_types=seq_types,
                max_new_ids=max_length, 
                temperature=temperature, 
                top_k=top_k
            )

        import pickle
        pickle.dump(output_seqs[1][0].cpu().numpy(), open("_zz.pkl", "wb"))

        import soundfile
        soundfile.write(file="_zz.wav", data=audio.cpu().numpy()[0,0], samplerate=sr)
        
        print(tokenizer.tok.convert_ids_to_tokens(output_seqs[1][0].cpu().numpy())) 

        from IPython import embed; embed(using=False); os._exit(0)
        
        print("------------")
        print("Audio path: {}".format(audio_path))
        for caption in captions:
            print("Ground truth: {}".format(caption))
        # soundfile.write(file="_zz.wav", data=audio[0][0].cpu().numpy(), samplerate=sr)
        
        # Extract audio embeddings
        audio_latent = get_audio_latent(
            model_name=audio_encoder_name, 
            model=audio_encoder, audio=audio
        )

        # Sample    
        for n in range(num_samples):

            # Combine audio embeddings and text ids
            input_seqs = [audio_latent, text_ids]
            seq_types = ["audio", "text"]

            with torch.no_grad():
                llm_decoder.eval()
            
                outputs = llm_decoder.generate(
                    seqs=input_seqs,
                    seq_types=seq_types,
                    max_new_tokens=max_length, 
                    temperature=temperature, 
                    top_k=top_k
                )
                # list of Tensor

            sampled_text_ids = outputs[-1][0].cpu().numpy()
            strings = tokenizer.decode(token_ids=sampled_text_ids, skip_special_tokens=True)
            print("Prediction: {}".format(strings))
            
        if audio_idx == 5:
            break


def get_clotho_meta(root: str, split: str) -> dict:
    r"""Load Clotho audio paths and captions."""
    if split == "train":
        meta_csv = Path(root, "clotho_captions_development.csv")
        audios_dir = Path(root, "clotho_audio_development")

    elif split == "test":
        meta_csv = Path(root, "clotho_captions_evaluation.csv")
        audios_dir = Path(root, "clotho_audio_evaluation")

    else:
        raise ValueError(split)

    meta_dict = {
        "audio_name": [],
        "audio_path": [],
        "captions": []
    }

    df = pd.read_csv(meta_csv, sep=',')

    for n in range(len(df)):
        meta_dict["audio_name"].append(df["file_name"][n])
        meta_dict["audio_path"].append(Path(audios_dir, df["file_name"][n]))
        meta_dict["captions"].append([df["caption_{}".format(i)][n] for i in range(1, 6)])

    return meta_dict


def tokens_to_string(tokens, tokenizer):
    return "".join([tokenizer.itos(token) for token in tokens])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)