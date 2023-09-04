
if [ ! -d ../bmodel ]; then
  mkdir ../bmodel
fi

if [ ! -d ../onnx ]; then
  mkdir ../onnx
fi


for seq_len in 64
do

    if [ ! -d ../bmodel_$seq_len ]; then
        mkdir ../bmodel_$seq_len
    fi
    
    for i in $(seq 0 0)
    do
    python3 export_onnx.py $i $seq_len
    model_transform.py \
        --model_name glm_block_nokvcache_$i \
        --model_def ../onnx/glm_lite_$i.onnx \
        --input_shapes [[$seq_len,4096],[1,$seq_len],[1,$seq_len],[1,$seq_len,$seq_len]] \
        --input_types float32,int32,in32,float32 \
        --channel_format none \
        --mlir ../bmodel_$seq_len/glm_block_$i.mlir
   	#--test_input inputs/input_embed.npz,inputs/position_ids.npz,inputs/block_position_ids.npz,inputs/attention_mask.npz  
    
    
    model_deploy.py \
        --mlir ../bmodel_$seq_len/glm_block_$i.mlir \
        --quantize F16 \
        --chip bm1684x \
        --model ../bmodel_$seq_len/glm_block_$i.bmodel
    done
done
rm glm_block*

