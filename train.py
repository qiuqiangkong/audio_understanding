from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from typing_extensions import Literal
import wandb

from audio_understanding.utils import parse_yaml, LinearWarmUp, remove_padded_columns
from audio_understanding.data.samplers import InfiniteSampler
from audidata.collate.default import collate_fn


def train(args) -> None:

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train")
    test_dataset = get_dataset(configs, split="test")

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Audio encoder
    audio_encoder = get_audio_encoder(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)
    
    # Tokenizer for converting text into IDs and vice versa
    tokenizer = get_tokenizer(configs=configs)
    
    # LLM decoder
    llm = get_llm(
        configs=configs, 
        audio_latent_dim=audio_encoder.latent_dim, 
        vocab_size=len(tokenizer),
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # Learnable parameters
    params = get_learnable_params(configs, audio_encoder, llm)
    
    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=params
    )

    # Logger
    if wandb_log:
        wandb.init(project="audio_understanding", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Prepare audio, question, and answering
        audio, question, answering = get_audio_question_answering(data)
        # audio: (b, c, t), question: (b, t), answering: (b, t)

        # 1.2 Encode audio into latent
        audio = audio.to(device)
        audio_latent = audio_encoder.encode(audio=audio)  # shape: (b, t, d)

        # 1.3 Tokenize question text to IDs
        question_ids = tokenizer.texts_to_ids(
            texts=question, 
            fix_length=configs["max_question_len"]
        ).to(device)  # shape: (b, t)

        # 1.4 Tokenize answering text to IDs
        answering_ids = tokenizer.texts_to_ids(
            texts=answering, 
            fix_length=configs["max_answering_len"]
        ).to(device)  # shape: (b, t)

        # 1.5 Remove padded columns to speed up training
        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(
                ids=answering_ids, 
                pad_token_id=tokenizer.pad_token_id
            )

        # 1.6 Prepare inputs
        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        # ------ 2. Training ------
        # 2.1 Forward
        llm.train()
        output_seqs = llm(
            seqs=seqs,
            seq_types=seq_types,
            mask=None
        )  # list

        # 2.2 Prepare data for next ID prediction
        output_seqs = [seq[:, 0 : -1] for seq in output_seqs]
        target_seqs = [seq[:, 1 :] for seq in seqs]
        
        # 2.3 Loss
        loss = ce_loss(
            output_seqs=output_seqs, 
            target_seqs=target_seqs, 
            loss_types=loss_types,
            ignore_index=tokenizer.pad_token_id
        )
        
        # 2.4 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)
        
        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                audio_encoder=audio_encoder,
                tokenizer=tokenizer, 
                llm=llm
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                audio_encoder=audio_encoder,
                tokenizer=tokenizer, 
                llm=llm
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            ckpt = {}
            
            if configs["audio_encoder"]["trainable"]:
                ckpt["audio_encoder"] = audio_encoder.state_dict()
            
            if configs["llm"]["trainable"]:
                ckpt["llm"] = llm.state_dict()

            torch.save(ckpt, ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break

        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from audidata.io.crops import RandomCrop, StartCrop
    from audidata.transforms import Mono, TimeShift, TextNormalization

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    datasets_split = "{}_datasets".format(split)

    datasets = []
    
    for name in configs[datasets_split].keys():
    
        if name == "GTZAN":

            from audio_understanding.datasets.gtzan import GTZAN

            dataset = GTZAN(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration), 
                transform=Mono(),
            )
            datasets.append(dataset)

        elif name == "LibriSpeech":

            from audio_understanding.datasets.librispeech import LibriSpeech

            dataset = LibriSpeech(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration), 
                transform=[Mono(), TimeShift(sr=sr, shift=(0., 0.5))],
            )
            datasets.append(dataset)

        elif name == "Clotho":

            from audio_understanding.datasets.clotho import Clotho

            dataset = Clotho(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=[Mono(), TimeShift(sr=sr, shift=(0., 0.5))],
                target_transform=TextNormalization()
            )
            datasets.append(dataset)

        elif name == "MAESTRO":

            from audio_understanding.datasets.maestro import MAESTRO
            from audio_understanding.target_transforms.midi import MIDI2Tokens
            from audidata.transforms.midi import PianoRoll

            CLS = locals()[configs["midi_to_tokens"]]
            
            dataset = MAESTRO(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1),
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                target_transform=[PianoRoll(fps=100, pitches_num=128), CLS(fps=configs["fps"])],
            )
            datasets.append(dataset)

        elif name == "AudioCaps":

            from audio_understanding.datasets.audiocaps import AudioCaps

            dataset = AudioCaps(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=TextNormalization()
            )
            datasets.append(dataset)

        elif name == "WavCaps":

            from audio_understanding.datasets.wavcaps import WavCaps

            dataset = WavCaps(
                root=configs[datasets_split][name]["root"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=TextNormalization()
            )
            datasets.append(dataset)

        else:
            raise ValueError(name)

    if len(datasets) == 1:
        return datasets[0]

    else:
        raise ValueError("Do not support multiple datasets in this file.")


def get_audio_encoder(configs: dict, ckpt_path: str) -> nn.Module:
    r"""Load pretrained audio encoder."""

    name = configs["audio_encoder"]["name"]
    sr = configs["sample_rate"]
    trainable = configs["audio_encoder"]["trainable"]

    if name == "Whisper":
        from audio_understanding.audio_encoders.whisper import Whisper
        model = Whisper(sr=sr, trainable=trainable)

    elif name == "PianoTranscriptionCRnn":
        from audio_understanding.audio_encoders.piano_transcription_crnn import PianoTranscriptionCRnn
        model = PianoTranscriptionCRnn(sr=sr, trainable=trainable)

    elif name == "PannsCnn14":
        from audio_understanding.audio_encoders.panns import PannsCnn14
        model = PannsCnn14(sr=sr, trainable=trainable)

    else:
        raise ValueError(name)

    if ckpt_path and configs["audio_encoder"]["trainable"]:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["audio_encoder"])

    return model


def get_tokenizer(configs: dict) -> nn.Module:
    r"""Get tokenizer."""

    name = configs["tokenizer"]["name"]

    if name == "Bert":
        from audio_understanding.tokenizers.bert import Bert
        tokenizer = Bert()

    elif name == "BertMIDI":
        from audio_understanding.tokenizers.bert_midi import BertMIDI
        tokenizer = BertMIDI()

    else:
        raise ValueError(name)

    return tokenizer


def get_llm(
    configs: dict, 
    audio_latent_dim: int, 
    vocab_size: int, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["llm"]["name"]

    if name == "Llama":

        from audio_understanding.llm.llama import Llama, LlamaConfig

        block_size = configs["llm"]["block_size"]
        n_layer = configs["llm"]["n_layer"]
        n_head = configs["llm"]["n_head"]
        n_embd = configs["llm"]["n_embd"]

        config = LlamaConfig(
            block_size=block_size,
            audio_latent_dim=audio_latent_dim, 
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd
        )
        model = Llama(config=config)

    else:
        raise ValueError(name)    

    if ckpt_path and configs["llm"]["trainable"]:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["llm"])

    return model


def get_learnable_params(
    configs: dict, 
    audio_encoder: nn.Module, 
    llm: nn.Module
) -> list:

    params = []

    if configs["audio_encoder"]["trainable"]:
        params += list(audio_encoder.parameters())

    if configs["llm"]["trainable"]:
        params += list(llm.parameters())

    return params


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler
        

def get_audio_question_answering(
    data: dict
) -> tuple[torch.Tensor, list[str], list[str]]:
    r"""Process data to audio, question, and answering according to different 
    datasets.

    Returns:
        audio: (b, c, t)
        question: (b, t)
        answering: (b, t)
    """

    name = data["dataset_name"][0]

    if name in ["GTZAN"]:
        return data["audio"], data["question"], data["label"]

    elif name in ["AudioCaps", "Clotho", "LibriSpeech", "WavCaps"]:
        return data["audio"], data["question"], data["caption"]

    elif name in ["MAESTRO"]:
        return data["audio"], data["question"], data["token"]

    else:
        raise ValueError(name)


def ce_loss(
    output_seqs: list[torch.Tensor], 
    target_seqs: list[torch.Tensor],
    loss_types: list[callable],
    ignore_index: int
) -> torch.float:
    r"""Calculate loss."""

    seqs_len = len(target_seqs)
    total_loss = 0.

    for i in range(len(output_seqs)):

        if loss_types[i] is None:
            continue

        elif loss_types[i] == "ce":
            total_loss += F.cross_entropy(
                input=output_seqs[i].flatten(0, 1),  # shape: (b*t, vocab_size)
                target=target_seqs[i].flatten(0, 1),  # shape: (b*t,)
                ignore_index=-1
            )

        else:
            raise ValueError(loss_types[i])

    return total_loss


def validate(
    configs: dict,
    dataset: Dataset,
    audio_encoder: nn.Module, 
    tokenizer: object,
    llm: nn.Module,
    valid_steps=50
) -> float:
    r"""Validate the model on part of data."""

    device = next(audio_encoder.parameters()).device
    losses = []

    batch_size = configs["train"]["batch_size_per_device"]
    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
        print("{}/{}".format(idx, len(dataset)))

        # ------ 1. Data preparation ------
        # 1.0 Collate data to batch
        data = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        data = collate_fn(data)

        # 1.1 Prepare audio, question, and answering
        audio, question, answering = get_audio_question_answering(data)
        # audio: (b, c, t), question: (b, t), answering: (b, t)

        # 1.3 Tokenize question text to IDs
        audio = audio.to(device)
        audio_latent = audio_encoder.encode(audio=audio, train_mode=False)  # shape: (b, t, d)

        # 1.4 Tokenize answering text to IDs
        question_ids = tokenizer.texts_to_ids(
            texts=question, 
            fix_length=configs["max_question_len"]
        ).to(device)  # shape: (b, t)

        # 1.5 Remove padded columns to speed up training
        answering_ids = tokenizer.texts_to_ids(
            texts=answering, 
            fix_length=configs["max_answering_len"]
        ).to(device)  # shape: (b, t)

        # 1.6 Prepare inputs
        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(
                ids=answering_ids, 
                pad_token_id=tokenizer.pad_token_id
            )

        # Prepare inputs
        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        # ------ 2. Training ------
        # 2.1 Forward
        with torch.no_grad():
            llm.eval()
            output_seqs = llm(
                seqs=seqs,
                seq_types=seq_types,
                mask=None
            )  # list

        # 2.2 Prepare data for next ID prediction
        output_seqs = [seq[:, 0 : -1] for seq in output_seqs]
        target_seqs = [seq[:, 1 :] for seq in seqs]
        
        # 2.3 Loss
        loss = ce_loss(
            output_seqs=output_seqs, 
            target_seqs=target_seqs, 
            loss_types=loss_types,
            ignore_index=tokenizer.pad_token_id
        )

        losses.append(loss.item())
        
    return np.mean(losses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)