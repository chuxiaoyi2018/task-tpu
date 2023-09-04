import sys
import numpy as np
from tqdm import tqdm
from tpu_perf.infer import SGInfer
from transformers import AutoTokenizer, GPT2Model
from time import time
import gc

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_v2(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def linear_norm(x):
    return x / np.sum(x)

class Pipeline:
    def __init__(
            self, seq_len
    ):
        # models
        self.model_nokvcache_dict = dict()
        self.model_kvcache_dict = dict()
        self.model_nokvcache_list = []
        self.model_kvcache_list = []

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


    def load_model(self, seq_len):
        # load embedding
        self.embedding_model = SGInfer('../embedding_lmhead_bmodel/embedding_f16.bmodel', devices=[12])

        # load finallayernorm & lmhead
        self.lmhead_model = SGInfer('../embedding_lmhead_bmodel/lm_head_f16.bmodel', devices=[12])

        # load nokvcache
        for i in range(28):
            block = SGInfer(f'../bmodel_{sys.argv[1]}/glm_block_{i}.bmodel', devices=[13])
            self.model_nokvcache_list.append(block)
        
        # 
        # del self.model_nokvcache_list
        # gc.collect()

        # load kvcache
        for i in range(14):
            block = SGInfer(f'../bmodel_{sys.argv[1]}/glm_block_kvcache_{i}.bmodel', devices=[14])
            self.model_kvcache_list.append(block)
        for i in range(14,28):
            block = SGInfer(f'../bmodel_{sys.argv[1]}/glm_block_kvcache_{i}.bmodel', devices=[14])
            self.model_kvcache_list.append(block)

    # convert string to input ids, add 130001 130004 in tail
    def preprocess_input(self, input_string):
        tokens = self.tokenizer.tokenize(input_string)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids += [130001, 130004]
        return input_ids

    # get nokvcache model input 
    def preprocess_with_nokvcache(self, input_ids, seq_len):
        valid_seq_len = len(input_ids)
        self.valid_seq_len = valid_seq_len
        diff_seq = seq_len - valid_seq_len

        # padding
        input_ids = np.array(input_ids + [0]*diff_seq if diff_seq > 0 else input_ids, dtype=np.int32)
        position_ids = np.array([[i for i in range(valid_seq_len-1)] + [valid_seq_len-2]*(diff_seq+1)], dtype=np.int32)
        block_position_ids = np.array([[0]*(valid_seq_len-1) + [1]*(diff_seq+1)], dtype=np.int32)

        # mask
        attention_mask = np.zeros((1,seq_len,seq_len)).astype(np.float32)
        attention_mask[:,:,valid_seq_len-1:] = 1
        attention_mask[:,valid_seq_len:,:] = 1
        attention_mask[:,valid_seq_len-1,valid_seq_len-1] = 0
        return input_ids, position_ids, block_position_ids, attention_mask

    # get kvcache model input
    def preprocess_with_kvcache(self, input_ids, span):
        # padding
        input_ids = np.array([input_ids[-1]], dtype=np.int32)
        block_position_ids = np.array([[span]], dtype=np.int32)
        return input_ids, block_position_ids, attention_mask

    def glm_forward(self, model, input_embed, position_ids, block_position_ids, attention_mask):
        begin_time = time()
        task_id = model.put(input_embed, position_ids, block_position_ids, attention_mask)
        task_id, results, valid = model.get()
        return results[0], results[1], results[2], time() - begin_time
    
    def glm_forward_with_kvcache(self, model, input_embed, position_ids, block_position_ids, past_key, past_value, attention_mask):
        begin_time = time()
        task_id = model.put(input_embed, position_ids, block_position_ids, attention_mask)
        task_id, results, valid = model.get()
        return results[0], results[1], results[2], time() - begin_time

    def embedding_forward(self, input_ids):
        begin_time = time()
        task_id = self.embedding_model.put(input_ids)
        task_id, results, valid = self.embedding_model.get()
        return results[0], time() - begin_time

    def lmhead_forward(self, hidden_states):
        begin_time = time()
        task_id = self.lmhead_model.put(hidden_states)
        task_id, results, valid = self.lmhead_model.get()
        return results[0], time() - begin_time

    def model_runner_with_nokvcache(self, input_ids, position_ids, block_position_ids, attention_mask, seq_len):
        total_time = 0
        key_list, value_list = [], []

        # embedding
        hidden_states, time = self.embedding_forward(input_ids)
        total_time += time

        # 28 block
        for model in self.model_nokvcache_list:
            hidden_states,key,value,time = self.glm_forward(model, hidden_states, position_ids, block_position_ids, attention_mask)
            key_list.append(key)
            value_list.append(value)
            total_time += time

        # finallayernorm & lmhead
        hidden_states = hidden_states[self.valid_seq_len-1:self.valid_seq_len]
        hidden_states, time = self.lmhead_forward(hidden_states)
        total_time += time

            
        #print("Forward Time:", total_time)
        return hidden_states, key_list, value_list

    def model_runner_with_kvcache(self, input_ids, position_ids, block_position_ids, attention_mask, past_key, past_value, seq_len):
        total_time = 0
        key_list, value_list = [], []

        # embedding
        hidden_states, time = self.embedding_forward(input_ids)
        total_time += time

        # 28 block
        for i, model in enumerate(self.model_kvcache_list):
            hidden_states,key,value,time = self.glm_forward(model, hidden_states, position_ids, block_position_ids, past_key[i], past_value[i], attention_mask)
            key_list.append(key)
            value_list.append(value)
            total_time += time

        # finallayernorm & lmhead
        hidden_states = hidden_states[self.valid_seq_len-1:self.valid_seq_len]
        hidden_states, time = self.lmhead_forward(hidden_states)
        total_time += time

        #print("Forward Time:", total_time)
        return hidden_states, key_list, value_list
    
    def postprocess(self, hidden_states, mode='greed', topk=10):
        if mode == 'greed':
            # next_token = self.tokenizer.decode(np.argmax(hidden_states))
            next_token = np.argmax(hidden_states)
        elif mode == 'sample':
            topk_index = np.argpartition(hidden_states[0],-topk)[-topk:]
            topk_logits = hidden_states[0, topk_index]
            topk_probs = softmax_v2(topk_logits)
            next_token = np.random.choice(topk_index, p=topk_probs, size=(1,))
        return next_token

    def model_runner(self, input_string, seq_len):
        total_time = 0
        input_ids = self.preprocess_input(input_string)

        # fix postion ids and attention mask for kvcache
        kvcache_position_ids = np.array([[len(input_ids) - 2]], dtype=np.int32)
        kvcache_attention_mask = np.zeros((1,seq_len,seq_len)).astype(np.float32)
        print(input_string + " ", flush=True, end="")


        for i in range(10):
            if i == 0:
                # init & nokvcache
                model_input_ids, position_ids, block_position_ids, attention_mask = self.preprocess_with_nokvcache(input_ids, seq_len)
                hidden_states, key_list, value_list = self.model_runner_with_nokvcache(model_input_ids, position_ids, block_position_ids, attention_mask, seq_len)
            else:
                # kvcache
                model_input_ids, block_position_ids = self.preprocess_with_kvcache(input_ids, i+1)
                hidden_states, key_list, value_list = self.model_runner_with_kvcache(model_input_ids, kvcache_position_ids, block_position_ids, key_list, value_list, kvcache_attention_mask, seq_len)
            next_token = self.postprocess(hidden_states)
            input_ids.append(next_token)
            import pdb;pdb.set_trace()
            output_string = self.tokenizer.decode(input_ids[-2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) 
            print(output_string[-1] + " ", flush=True, end="")
            
        return hidden_states

begin_time = time()

# preprocess
seq_len = int(sys.argv[1])
print(f"===================================================={seq_len}====================================================")
input_string = "i love you"


pipeline = Pipeline(seq_len)
pipeline.load_model(seq_len)
pipeline.model_runner(input_string, seq_len)

print("Total Time", time() - begin_time)

