model_transform.py     --model_name uie_nano_pytorch     --model_def uie_nano_pytorch/inference.onnx     --input_shapes [[1,512],[1,512],[1,512]]  --mlir tmp/uie_nano_pytorch.mlir

model_deploy.py   --mlir tmp/uie_nano_pytorch.mlir  --quantize F16     --chip bm1684x     --model tmp/uie_nano_pytorch_f16.bmodel

