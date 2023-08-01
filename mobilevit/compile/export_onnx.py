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
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests
import pdb
import numpy as np
import functools
import operator

MODEL_PATH = "apple/mobilevit-small"
folder = "./tmp"

origin_model = MobileViTForImageClassification.from_pretrained(MODEL_PATH)

origin_model.eval()

def convert_mobilevit(model):
    # input
    batch_size = 1
    
    inputs = torch.randn((batch_size, 3, 256, 256))

    torch.onnx.export(
        model,
        (inputs),
        f'{folder}/mobilevit.onnx',
        export_params=True,
        input_names=['inputs'],
        output_names=['outputs'],
    )


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
convert_mobilevit(origin_model)
