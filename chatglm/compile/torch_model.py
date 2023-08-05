import sys
from transformers import AutoTokenizer, AutoModel


CHATGLM2_PATH = sys.argv[1]

model = AutoModel.from_pretrained(CHATGLM2_PATH, trust_remote_code=True).float()

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
import pdb;pdb.set_trace()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

