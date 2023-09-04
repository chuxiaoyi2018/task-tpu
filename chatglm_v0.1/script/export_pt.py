import torch
  
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
from time import sleep;sleep(5)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
#tokenizer = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/1b54948bb28de5258b55b893e193c3046a0b0484/")
#model = AutoModel.from_pretrained("/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/1b54948bb28de5258b55b893e193c3046a0b0484/").float()

for i in tqdm(range(28)):
    torch.save(model.transformer.layers[i].state_dict(), f'../block_pt/block_{i}.pt')
