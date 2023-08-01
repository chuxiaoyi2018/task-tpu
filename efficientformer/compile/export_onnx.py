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
import torch
import timm

origin_model = timm.create_model('efficientformer_l1.snap_dist_in1k', pretrained=True)
origin_model = origin_model.eval()

folder = "./tmp"

def convert_efficientformer(model):
    # input
    batch_size = 1
    
    inputs = torch.randn((batch_size, 3, 224, 224))

    torch.onnx.export(
        model,
        (inputs),
        f'{folder}/efficientformer_v1.onnx',
        export_params=True,
        input_names=['inputs'],
        output_names=['outputs'],
    )


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
convert_efficientformer(origin_model)
