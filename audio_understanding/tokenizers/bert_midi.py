from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer

from audio_understanding.utils import pad_or_truncate


class BertMIDI:
    r"""Extend text tokenizer with discrete audio codec vocabularies.
    """
    
    def __init__(self) -> None:

        super().__init__()

        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        new_vocabs = []

        new_vocabs += ["time_index={}".format(t) for t in range(6001)]
        new_vocabs += ["name=note_onset", "name=note_sustain", "name=note_offset"]
        new_vocabs += ["name=pedal_onset", "name=pedal_sustain", "name=pedal_offset"]

        new_vocabs += ["pitch={}".format(p) for p in range(128)]
        new_vocabs += ["velocity={}".format(v) for v in range(128)]
        new_vocabs += ["program={}".format(p) for p in range(128)]

        # Merge text tokens and audio tokens
        print("Original vocab size: {}".format(len(self.tok)))
        print("New vocab size: {}".format(len(new_vocabs)))
        self.tok.add_tokens(new_vocabs)
        print("Extended vocab size: {}".format(len(self.tok)))

    def texts_to_ids(
        self,
        texts: list[str] | list[list[str]], 
        fix_length: int
    ) -> torch.LongTensor:
        r"""Convert texts to IDs. 

        E.g., ["Hello world"]
           -> [[101, 8667, 1362, 102, 0, 0]]

        Args:
            texts: list[str] | list[list[str]]
            fix_length: int

        Returns:
            batch_ids: (b, t)
        """

        batch_ids = []

        for text in texts:
        
            # Convert texts to tokens, e.g., "Hello world." -> ["Hello", "world", "."]
            if isinstance(text, str):
                tokens = self.tok.tokenize(text)
            elif isinstance(text, list):
                tokens = text
            else:
                raise TypeError(text)

            # Convert tokens to IDs. Reserve 2 IDs for special IDs
            ids = self.tok.convert_tokens_to_ids(tokens)[0 : fix_length - 2]

            assert ids.count(self.tok.unk_token_id) == 0, "Unknown token is not allowed! Please extend the vocabulary!"

            # Append special IDs
            ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]

            # Pad
            if fix_length:
                ids = pad_or_truncate(ids, fix_length, self.tok.pad_token_id)

            batch_ids.append(ids)

        return torch.LongTensor(batch_ids)

    def __len__(self):
        return len(self.tok)

    @property
    def cls_token_id(self):
        return self.tok.cls_token_id

    @property
    def pad_token_id(self):
        return self.tok.pad_token_id

    @property
    def boa_token_id(self):
        return self.tok.convert_tokens_to_ids("<boa>")

    @property
    def eoa_token_id(self):
        return self.tok.convert_tokens_to_ids("<eoa>")