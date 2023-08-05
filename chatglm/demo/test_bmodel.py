import numpy as np
from tpu_perf.infer import SGInfer

for i in range(28):
    model = SGInfer(f"/workspace/task-tpu/customer_data/compile/tmp/glm_block/glm_block_{i}.bmodel", devices=[14])
    embedding = SGInfer("/workspace/task-tpu/customer_data/compile/tmp/embedding/embedding_512_f16.bmodel", devices=[14])
    
    input_ids = np.array([9651, 26762, 130001, 130004] + [0] * 508, dtype=np.int32)
    
    task_id = embedding.put(input_ids)
    task_id, results, valid = embedding.get()
    
    input_embed = results[0].reshape(512,1,4096)
    position_len = 2
    MAX_LEN = 512
    token_len = 4
    span_len = 1
    position_ids = list(range(position_len)) + 2 * [position_len] + (MAX_LEN - token_len) * [0]
    block_position_ids = (token_len - 1) * [0] + [span_len] + (MAX_LEN - token_len) * [0]
    position_ids = np.array([[position_ids, block_position_ids]])
    attention_mask = np.ones((1,1,MAX_LEN, MAX_LEN))
    
    attention_mask[0,0,0,0] = 0
    attention_mask[0,0,1,:2] = 0
    attention_mask[0,0,2,:3] = 0
    attention_mask[0,0,3,:4] = 0
    
    attention_mask = attention_mask.astype(np.float32)
    
    position_ids = position_ids.astype(np.int32)
    task_id = model.put(input_embed, position_ids, attention_mask)
    task_id, results, valid = model.get()
    
    print(np.isnan(results[0]).any()) 



