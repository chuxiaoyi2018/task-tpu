model_transform.py     --model_name decoder     --model_def tmp/opus-mt-zh-en-decoder.onnx     --input_shapes [[1,1],[1,1],[1,1,512],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64],[1,8,511,64]]     --mlir tmp/opus-mt-zh-en-decoder.mlir 

model_transform.py     --model_name encoder     --model_def tmp/opus-mt-zh-en-encoder.onnx     --input_shapes [[1,511],[1,511]]  --mlir tmp/opus-mt-zh-en-encoder.mlir

model_transform.py     --model_name init-decoder  --model_def tmp/opus-mt-zh-en-init-decoder.onnx     --input_shapes [[1,511],[1,511],[1,511,512]]  --mlir tmp/opus-mt-zh-en-init-decoder.mli


model_deploy.py   --mlir tmp/opus-mt-zh-en-decoder.mlir  --quantize F16     --chip bm1684x     --model tmp/opus-mt-zh-en-decoder_f16.bmodel

model_deploy.py   --mlir tmp/opus-mt-zh-en-encoder.mlir  --quantize F16     --chip bm1684x     --model tmp/opus-mt-zh-en-encoder_f16.bmodel

model_deploy.py   --mlir tmp/opus-mt-zh-en-init-decoder.mlir  --quantize F16     --chip bm1684x     --model tmp/opus-mt-zh-en-init-decoder_f16.bmodel