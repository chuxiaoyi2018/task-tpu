#!/bin/bash
set -ex

pushd tmp
model_transform.py \
--model_name  unet \
--model_def unet_scale0.5.onnx \
--input_shapes [[1,3,572,572]] \
--pixel_format rgb \
--mean 0.0,0.0,0.0 \
--scale 0.0039216,0.0039216,0.0039216 \
--mlir unet_scale0.5.mlir;

model_deploy.py \
--mlir unet_scale0.5.mlir \
--quantize F32 \
--chip bm1684x \
--tolerance 0.99,0.99 \
--model unet_scale0.5_f32.bmodel;

run_calibration.py unet_scale0.5.mlir \
   --dataset ../../../data/caliset \
   --input_num 1 \
   -o unet_scale0.5_cali_table

model_deploy.py \
--mlir unet_scale0.5.mlir \
--quantize INT8 \
--calibration_table unet_scale0.5_cali_table \
--chip bm1684x \
--tolerance 0.80,0.80 \
--model unet_scale0.5_int8.bmodel;

popd
