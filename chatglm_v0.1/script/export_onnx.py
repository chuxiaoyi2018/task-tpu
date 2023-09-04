import torch
import math
import sys
from configuration_chatglm import ChatGLMConfig
from torch.nn import LayerNorm
from torch.nn.utils import skip_init
from modeling_chatglm import GLMBlock

config = ChatGLMConfig()
params_dtype = torch.half
config.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

block = GLMBlock(
        config.hidden_size,
        config.num_attention_heads,
        config.layernorm_epsilon,
        layer_id=0,
        inner_hidden_size=config.inner_hidden_size,
        hidden_size_per_attention_head=config.hidden_size_per_attention_head,
        layernorm=LayerNorm,
        use_bias=True,
        params_dtype=params_dtype,
        position_encoding_2d=config.position_encoding_2d,
    ).float()


# # use float() to avoid float16 block weight 
block.load_state_dict(torch.load(f"../block_pt/block_{sys.argv[1]}.pt"), strict=True)
block.eval()
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", revision="v1.1.0", trust_remote_code=True).float()
# block = model.transformer.layers[int(sys.argv[1])]
block.eval()



class Model_lite(torch.nn.Module):
    def __init__(self, block, seq_len, cos_cached, sin_cached, layer_id):
        super().__init__()
        # params
        self.seq_len = seq_len
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached
        
        self.alpha = (2 * config.num_layers) ** 0.5
        
        self.query_key_layer_scaling_coeff = float(layer_id + 1)
        self.coeff = (math.sqrt(128) * query_key_layer_scaling_coeff)
        self.block = block

    def forward(self, hidden_states, position_ids, block_position_ids, attention_mask):
        # position ids
        pos_cos = torch.nn.functional.embedding(position_ids, self.cos_cached.squeeze(1)).transpose(0,1)
        pos_sin = torch.nn.functional.embedding(position_ids, self.sin_cached.squeeze(1)).transpose(0,1)
        # block ids
        block_cos = torch.nn.functional.embedding(block_position_ids, self.cos_cached.squeeze(1)).transpose(0,1)
        block_sin = torch.nn.functional.embedding(block_position_ids, self.sin_cached.squeeze(1)).transpose(0,1)

        # input layernorm
        residual = self.block.input_layernorm(hidden_states) # [512,4096]
        # self attention
        hidden_states =  torch.matmul(residual, self.block.attention.query_key_value.weight.transpose(1,0)) + self.block.attention.query_key_value.bias
        hidden_states = hidden_states.reshape(self.seq_len,32,384) # [512,32,384]
        query, key, value = hidden_states.split(128,dim=2)
        
        # rotary emb
        q1, q2 = query.split(64, dim=2) # [512,32,64]
        k1, k2 = key.split(64, dim=2)
        
        q1_half = torch.cat((-q1[:,:,32:], q1[:,:,:32]), dim=-1)
        k1_half = torch.cat((-k1[:,:,32:], k1[:,:,:32]), dim=-1)
        q1, k1 = (q1 * pos_cos) + (q1_half * pos_sin), (k1 * pos_cos) + (k1_half * pos_sin)
        
        q2_half = torch.cat((-q2[:,:,32:], q2[:,:,:32]), dim=-1)
        k2_half = torch.cat((-k2[:,:,32:], k2[:,:,:32]), dim=-1)
        q2, k2 = (q2 * block_cos) + (q2_half * block_sin), (k2 * block_cos) + (k2_half * block_sin)
        
        q, k = torch.cat((q1, q2), dim=-1), torch.cat((k1, k2), dim=-1)
        
        # compute attention score
        attention_scores = torch.matmul(q.transpose(0,1)/self.coeff, k.transpose(0,1).transpose(1,2)) # [32,512,512]
        attention_scores.masked_fill_(attention_mask, -100.0)
        attention_scores = attention_scores * self.query_key_layer_scaling_coeff
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=2) # [32,512,512]
        
        hidden_states = torch.bmm(attention_probs, value.transpose(0,1)) # [32,512,512] * [32,512,128]=[32,512,128] 误差似乎是在这里体现
        hidden_states = hidden_states.transpose(0,1) # [512,32,128]
        hidden_states = hidden_states.reshape(self.seq_len,4096) # [512,4096]

        hidden_states = self.block.attention.dense(hidden_states)
        # hidden_states = torch.matmul(hidden_states,self.block.attention.dense.weight.transpose(1,0))+self.block.attention.dense.bias
        hidden_states = residual * self.alpha + hidden_states
        
        # MLP
        residual = self.block.post_attention_layernorm(hidden_states)
        hidden_states = self.block.mlp(residual)
        hidden_states = residual * self.alpha + hidden_states

        return hidden_states

seq_len = int(sys.argv[2])
# compute cos, sin
inv_freq = block.attention.rotary_emb.inv_freq
t = torch.arange(seq_len-1)
freqs = torch.einsum('i,j->ij', t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos_cached = emb.cos()[:, None, :]
sin_cached = emb.sin()[:, None, :]
cos_cached, sin_cached = cos_cached[:seq_len, ...], sin_cached[:seq_len, ...]

alpha = (2 * config.num_layers) ** 0.5

layer_id = int(sys.argv[1])
query_key_layer_scaling_coeff = float(layer_id + 1)
coeff = (math.sqrt(128) * query_key_layer_scaling_coeff)



# input
torch.manual_seed(42)
input_embed = torch.randn((seq_len, 4096))
position_ids = torch.tensor([[i for i in range(seq_len-1)] + [seq_len-2]])
block_position_ids = torch.tensor([[0]*(seq_len-1) + [1]])
attention_mask = torch.zeros(1,seq_len,seq_len,dtype=torch.bool)

# attention mask
attention_mask[:,:,-1] = True
attention_mask[:,-1,-1] = False

model = Model_lite(block, seq_len, cos_cached, sin_cached, layer_id)
hidden_states = model.forward(input_embed, position_ids, block_position_ids, attention_mask)

torch.onnx.export(model,
                  (input_embed, position_ids, block_position_ids, attention_mask),
                  f'../onnx/glm_lite_{sys.argv[1]}.onnx',
                  verbose=False, input_names=['input_embed', 'position_ids', 'block_position_ids', 'attention_mask'],
                  output_names=['hidden_states'],
                  opset_version=15
                  )


