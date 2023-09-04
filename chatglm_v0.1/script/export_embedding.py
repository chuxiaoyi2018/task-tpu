import torch

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
from time import sleep;sleep(5)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
#tokenizer = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/1b54948bb28de5258b55b893e193c3046a0b0484/")
#model = AutoModel.from_pretrained("/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/1b54948bb28de5258b55b893e193c3046a0b0484/").float()
model.eval()

class Model_lite(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # params
        self.model = model

    def forward(self, hidden_states):
        hidden_states = self.model.transformer.word_embeddings(hidden_states)
        return hidden_states

 
model_lite = Model_lite(model)
for param in model_lite.parameters():
    param.requires_grad = False

hidden_states = torch.randn(1,64)[0].long() + 20
logits = model_lite.forward(hidden_states)

torch.onnx.export(model_lite,
                  (hidden_states),
                  f'../onnx/embedding.onnx',
                  verbose=False, 
                  input_names=['hidden_states'],
                  output_names=['logits'],
                  opset_version=15)

