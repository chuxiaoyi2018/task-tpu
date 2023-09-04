
pip3 install https://github.com/sophgo/tpu-perf/releases/download/v1.2.10/tpu_perf-1.2.10-py3-none-manylinux2014_x86_64.whl
pip install sentencepiece transformers==4.27.1 cpm_kernels protobuf==3.20.3

if [ ! -d ../block_pt ]; then
  mkdir ../block_pt
fi

