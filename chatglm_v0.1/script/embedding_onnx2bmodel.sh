model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[64]] \
    --channel_format none \
    --mlir ../embedding_lmhead_bmodel/embedding.mlir


model_deploy.py \
    --mlir ../embedding_lmhead_bmodel/embedding.mlir \
    --quantize F16 \
    --chip bm1684x \
    --tolerance 0.9,0.9 \
    --model ../embedding_lmhead_bmodel/embedding_f16.bmodel




