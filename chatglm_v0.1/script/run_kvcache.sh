
if [ ! -d ../bmodel ]; then
  mkdir ../bmodel
fi

if [ ! -d ../onnx ]; then
  mkdir ../onnx
fi


for seq_len in 64 128 256 512
do
    if [ ! -d ../bmodel_$seq_len ]; then
        mkdir ../bmodel_$seq_len
    fi
    
    for i in $(seq 0 27)
    do

    python3 export_onnx_kvcache.py $i $seq_len
    
    model_transform.py \
        --model_name glm_block_$i \
        --model_def ../onnx/glm_kvcache_$i.onnx \
        --input_shapes [[1,4096],[1,1],[1,1],[$seq_len,32,128],[$seq_len,32,128],[1,1,1]] \
        --input_types float32,int32,in32,float32,float32,bool \
        --channel_format none \
        --mlir ../bmodel_$seq_len/glm_block_$i.mlir
    
    
    
    model_deploy.py \
        --mlir ../bmodel_$seq_len/glm_block_$i.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model ../bmodel_$seq_len/glm_block_kvcache_$i.bmodel
    done

    dtools upload --name yi.chu --password Do467813@ --local_dir ../bmodel_$seq_len --nas_dir /home/model/chatglm/bmodel_$seq_len

done
rm glm_block*

