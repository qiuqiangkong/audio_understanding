from __future__ import annotations

import argparse
from pathlib import Path
import re

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from typing_extensions import Literal
import wandb

from audio_understanding.utils import LinearWarmUp, parse_yaml, remove_padded_columns
from audio_understanding.data.samplers import InfiniteSampler
from audidata.collate.default import collate_fn

from train import get_dataset, get_audio_encoder, get_tokenizer, get_llm, get_learnable_params, get_optimizer_and_scheduler, get_audio_question_answering, ce_loss, validate


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
    )
    
    # Tokenizer for converting text into IDs and vice versa
    tokenizer = get_tokenizer(configs=configs)
    
    # LLM decoder
    llm = get_llm(
        configs=configs, 
        audio_latent_dim=audio_encoder.latent_dim, 
        vocab_size=len(tokenizer),
        ckpt_path=configs["train"]["resume_ckpt_path"]
    )
    
    # Learnable parameters
    params = get_learnable_params(configs, audio_encoder, llm)
    
    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=params
    )

    # Prepare for acceleration
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    audio_encoder, llm, tokenizer, optimizer, train_dataloader = accelerator.prepare(
        audio_encoder, llm, tokenizer, optimizer, train_dataloader)

    # Logger
    if wandb_log and accelerator.is_main_process:
        wandb.init(project="audio_understanding", name="{}".format(config_name))

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Prepare audio, question, and answering
        audio, question, answering = get_audio_question_answering(data)
        # audio: (b, c, t), question: (b, t), answering: (b, t)

        # 1.2 Encode audio into latent
        audio_latent = audio_encoder.module.encode(audio=audio)  # shape: (b, t, d)
        device = audio_latent.device

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
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad

        # 2.5 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print(loss)
        
        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0 and accelerator.is_main_process:

            train_loss = validate(
                configs=configs,
                dataset=train_dataset, 
                audio_encoder=accelerator.unwrap_model(audio_encoder),
                tokenizer=accelerator.unwrap_model(tokenizer), 
                llm=accelerator.unwrap_model(llm)
            )

            test_loss = validate(
                configs=configs,
                dataset=test_dataset, 
                audio_encoder=accelerator.unwrap_model(audio_encoder),
                tokenizer=accelerator.unwrap_model(tokenizer), 
                llm=accelerator.unwrap_model(llm)
            )

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step
                )

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0 and accelerator.is_main_process:
            
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)