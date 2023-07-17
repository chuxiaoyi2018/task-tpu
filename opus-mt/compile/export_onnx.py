#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead
import pdb
import numpy as np
import functools
import operator

OPUSMT_PATH = "Helsinki-NLP/opus-mt-zh-en"
folder = "./tmp"

origin_model = AutoModelWithLMHead.from_pretrained(OPUSMT_PATH, trust_remote_code=True).float()


origin_model.eval()
config = origin_model.config
for param in origin_model.parameters():
    param.requires_grad = False
tokenizer = AutoTokenizer.from_pretrained(OPUSMT_PATH, trust_remote_code=True)


class DecoderWithLMhead(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias

    def forward(self, *inputs):
        input_ids, attention_mask, encoder_hidden_states = inputs[:3]

        list_pkv = inputs[3:]
        past_key_values = tuple(list_pkv[i:i+4]
                                for i in range(0, len(list_pkv), 4))
        decoder_output = self.decoder(input_ids=input_ids,  # decoder_input_ids
                                      encoder_attention_mask=attention_mask,
                                      encoder_hidden_states=encoder_hidden_states,
                                      past_key_values=past_key_values)
        lm_head_out = self.lm_head(decoder_output[0]) + self.final_logits_bias

        # move from 0 1 2 .. 511 to 1 2 3 ..511
        key_values = tuple((d[0][:,:,1:,:], d[1][:,:,1:,:]) for d in decoder_output[1])
        return lm_head_out, key_values


class Encoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]


class DecoderWithLMheadInitial(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, final_logits_bias):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        decoder_output = self.decoder(input_ids=input_ids,
                                      encoder_attention_mask=attention_mask,
                                      encoder_hidden_states=encoder_hidden_states)
        key_values = tuple((d[0][:,:,1:,:], d[1][:,:,1:,:]) for d in decoder_output[1])
        return self.lm_head(decoder_output[0]) + self.final_logits_bias, decoder_output[1]

def convert_encoder(simplified_encoder):
    input_ids = torch.LongTensor(1, config.max_length - 1).random_(0, config.vocab_size)
    attention_mask = torch.ones(1, config.max_length - 1)
    torch.onnx.export(
        simplified_encoder,
        (input_ids, attention_mask),
        f'{folder}/opus-mt-zh-en-encoder.onnx',
        export_params=True,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['hidden_states']
    )


def convert_decoder(decoder_with_lm_head):
    # input
    batch_size = 1
    n_heads = config.decoder_attention_heads
    seq_length_a, seq_length_b = 511, 511 # must be 511 not 512 to avoid index out of range
    d_kv = 64

    input_ids_dec = torch.ones((1, 1), dtype=torch.int64)
    attention_mask_dec = torch.ones((1, 1), dtype=torch.int64)
    enc_out = torch.ones(
        (batch_size, 1, config.d_model), dtype=torch.float32)

    a = torch.ones((batch_size, n_heads, seq_length_a, d_kv),
                   dtype=torch.float32)  # 1, 8, 511, 64
    b = torch.ones((batch_size, n_heads, seq_length_b, d_kv),
                   dtype=torch.float32)  # 1, 8, 511, 64
    block = (a, a, b, b)
    past_key_values = (block, ) * config.decoder_layers

    flat_past_key_values = functools.reduce(
        operator.iconcat, past_key_values, [])

    decoder_all_inputs = tuple(
        [input_ids_dec, attention_mask_dec, enc_out] + flat_past_key_values)

    decoder_inputs = ['input_ids',
                        'encoder_attention_mask', 'encoder_hidden_states']

    pkv_input_names = ['input_{}'.format(i)
                        for i in range(0, 24)]

    decoder_input_names = decoder_inputs + pkv_input_names

    decoder_output_names = ['logits', 'output_past_key_values']

    torch.onnx.export(
        decoder_with_lm_head,
        decoder_all_inputs,
        f'{folder}/opus-mt-zh-en-decoder.onnx',
        export_params=True,
        do_constant_folding=True,
        input_names=decoder_input_names,
        output_names=decoder_output_names,
    )

def convert_init_decoder(decoder_with_lm_head_init):
    # input
    batch_size = 1
    n_heads = config.decoder_attention_heads
    seq_length_a, seq_length_b = 511, 511
    d_kv = 64

    input_ids_dec = torch.ones((1, 511), dtype=torch.int64)
    attention_mask_dec = torch.ones((1, seq_length_b), dtype=torch.int64)
    enc_out = torch.ones(
        (batch_size, seq_length_b, config.d_model), dtype=torch.float32)

    torch.onnx.export(
        decoder_with_lm_head_init,
        (input_ids_dec, attention_mask_dec, enc_out),
        f'{folder}/opus-mt-zh-en-init-decoder.onnx',
        export_params=True,
        input_names=[
            'input_ids', 'encoder_attention_mask', 'encoder_hidden_states'],
        output_names=['logits', 'past_key_values'],
    )


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

encoder = Encoder(origin_model.model.encoder)
init_decoder = DecoderWithLMheadInitial(origin_model.model.decoder, 
                                        origin_model.lm_head, origin_model.final_logits_bias)
decoder = DecoderWithLMhead(origin_model.model.decoder, 
                            origin_model.lm_head, origin_model.final_logits_bias)

# export models
convert_encoder(encoder)
convert_init_decoder(init_decoder)
convert_decoder(decoder)
