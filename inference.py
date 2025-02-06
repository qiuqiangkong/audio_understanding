from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import librosa
import pretty_midi
import torch

from audio_understanding.utils import parse_yaml
from train import get_audio_encoder, get_llm, get_tokenizer


def inference(args) -> None:

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
    audio_latent = audio_encoder.encode(audio=audio, train_mode=False)  # shape: (b, t, d)

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
            temperature=temperature, 
            top_k=get_top_k(config_yaml)
        )

    # Get answering from output seqs
    answering_ids = output_seqs[2][0]  # shape: (t)

    answering_texts = convert_ids_to_texts(config_yaml, tokenizer, answering_ids)
    print("Output: {}".format(answering_texts))


def get_top_k(config_yaml: str) -> int:

    if Path(config_yaml).stem in ["asr_librispeech"]:
        return 1

    elif Path(config_yaml).stem in ["music_tagging_gtzan"]:
        return 1

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        return 1

    elif Path(config_yaml).stem in ["audio_caption_clotho"]:
        return 100

    else:
        print("Using default top_k=100")
        return 100

def get_question(config_yaml: str) -> str:
    
    if Path(config_yaml).stem in ["music_tagging_gtzan"]:
        return "Music tagging."

    elif Path(config_yaml).stem in ["asr_librispeech"]:
        return "Automatic speech recognition."

    elif Path(config_yaml).stem in ["audio_caption_clotho"]:
        return "Audio caption."

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        return "Music transcription."

    else:
        raise NotImplementedError("Users need to write a question!")


def convert_ids_to_texts(config_yaml: dict, tokenizer, answering_ids: list[int]):

    if Path(config_yaml).stem in ["asr_librispeech", "audio_caption_clotho", "music_tagging_gtzan"]:
        return tokenizer.tok.decode(answering_ids, skip_special_tokens=True)

    elif Path(config_yaml).stem in ["piano_transcription_maestro"]:
        tokens = tokenizer.tok.convert_ids_to_tokens(answering_ids)
        pickle.dump(tokens, open("tmp_tokens.pkl", "wb"))

        # Dump for speeding up debug
        tokens = pickle.load(open("tmp_tokens.pkl", "rb"))
        configs = parse_yaml(config_yaml)
        tokens_to_midi(tokens=tokens, fps=configs["fps"], output_path="output.mid")
        return tokens

    else:
        print("Using default ids to texts conversion.")
        return tokenizer.tok.decode(answering_ids, skip_special_tokens=True)


def tokens_to_midi(tokens: list[str], fps: float, output_path: str) -> None:

    note_dict = {pitch: [] for pitch in range(128)}
    tokens_num = len(tokens)

    for i in range(len(tokens)):

        print(i, tokens[i])

        if "=" in tokens[i]:

            key, value = tokens[i].split("=")

            if value == "note_onset" and i + 3 < tokens_num:
                key, value = tokens[i + 1].split("=")
                assert key == "time_index"
                time_index = int(value)

                key, value = tokens[i + 2].split("=")
                assert key == "pitch"
                pitch = int(value)

                key, value = tokens[i + 3].split("=")
                assert key == "velocity"
                velocity = int(value)

                note = {"onset_time_index": time_index, "pitch": pitch, "velocity": velocity}
                note_dict[pitch].append(note)

            elif value == "note_offset" and i + 2 < tokens_num:
                key, value = tokens[i + 1].split("=")
                assert key == "time_index"
                time_index = int(value)

                key, value = tokens[i + 2].split("=")
                assert key == "pitch"
                pitch = int(value)

                if len(note_dict[pitch]) > 0:
                    note_dict[pitch][-1]["offset_time_index"] = time_index

    events = []

    for pitch in note_dict.keys():
        events += note_dict[pitch]

    # Write out MIDI
    track = pretty_midi.Instrument(program=0)
    track.is_drum = False

    for e in events:
        start_time = e["onset_time_index"] / 100

        if "offset_time_index" in e.keys():
            end_time = e["offset_time_index"] / 100
        else:
            end_time = start_time + 0.1

        note = pretty_midi.Note(
            pitch=e["pitch"], 
            start=start_time, 
            end=end_time, 
            velocity=e["velocity"]
        )
        track.notes.append(note)

    midi_data = pretty_midi.PrettyMIDI()
    midi_data.instruments.append(track)
    midi_data.write(output_path)
    print("Write out to {}".format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)

    args = parser.parse_args()

    inference(args)