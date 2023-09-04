
for name in kvcache
do
    for seq_len in 64
        do
        tpu_model --combine ../bmodel_$seq_len/glm_block_${name}_0.bmodel ../bmodel_$seq_len/glm_block_${name}_1.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_2.bmodel ../bmodel_$seq_len/glm_block_${name}_3.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_4.bmodel ../bmodel_$seq_len/glm_block_${name}_5.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_6.bmodel ../bmodel_$seq_len/glm_block_${name}_7.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_8.bmodel ../bmodel_$seq_len/glm_block_${name}_9.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_10.bmodel ../bmodel_$seq_len/glm_block_${name}_11.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_12.bmodel ../bmodel_$seq_len/glm_block_${name}_13.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_14.bmodel ../bmodel_$seq_len/glm_block_${name}_15.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_16.bmodel ../bmodel_$seq_len/glm_block_${name}_17.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_18.bmodel ../bmodel_$seq_len/glm_block_${name}_19.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_20.bmodel ../bmodel_$seq_len/glm_block_${name}_21.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_22.bmodel ../bmodel_$seq_len/glm_block_${name}_23.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_24.bmodel ../bmodel_$seq_len/glm_block_${name}_25.bmodel \
                            ../bmodel_$seq_len/glm_block_${name}_26.bmodel ../bmodel_$seq_len/glm_block_${name}_27.bmodel \
                            -o ../combine/glm_block_combine_${seq_len}_${name}.bmodel
        done
done
