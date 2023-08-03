#!/bin/bash
set -ex

model='mobilevit'

pushd tmp
model_transform.py \
  --model_name  $model \
  --model_def $model.onnx \
  --input_shapes [[1,3,256,256]] \
  --pixel_format rgb \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --mlir $model.mlir

model_deploy.py \
  --mlir $model.mlir \
  --quantize F32 \
  --chip bm1684x \
  --tolerance 0.99,0.99 \
  --model $model\_f32.bmodel

run_calibration.py $model.mlir \
   --dataset ../../../data/caliset \
   --input_num 50 \
   -o $model\_cali_table

run_sensitive_layer.py $model.mlir \
   --dataset ../../../data/caliset \
   --input_num 25 \
   --inference_num 15 \
   --max_float_layers 40 \
   --expected_cos 0.90 \
   --calibration_table $model\_cali_table \
   --chip bm1684x\
   -o $model\_qtable


model_deploy.py \
    --mlir $model.mlir \
    --quantize INT8 \
    --quantize_table $model\_qtable \
    --calibration_table $model\_cali_table \
    --chip bm1684x \
    --tolerance 0.80,0.80 \
    --model $model\_int8.bmodel

popd
