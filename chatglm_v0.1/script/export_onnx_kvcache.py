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


# use float() to avoid float16 block weight 
block.load_state_dict(torch.load(f"../block_pt/block_{sys.argv[1]}.pt"), strict=True)
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
        self.coeff = (math.sqrt(128) * self.query_key_layer_scaling_coeff)
        self.block = block

    def forward(self, hidden_states, position_ids, block_position_ids, past_key, past_value, attention_mask):
        # position ids
        pos_cos = torch.nn.functional.embedding(position_ids, self.cos_cached.squeeze(1)).transpose(0,1)
        pos_sin = torch.nn.functional.embedding(position_ids, self.sin_cached.squeeze(1)).transpose(0,1)
        # block ids
        block_cos = torch.nn.functional.embedding(block_position_ids, self.cos_cached.squeeze(1)).transpose(0,1)
        block_sin = torch.nn.functional.embedding(block_position_ids, self.sin_cached.squeeze(1)).transpose(0,1)

        # input layernorm
        # import pdb;pdb.set_trace()
        residual = self.block.input_layernorm(hidden_states) # [512,4096]
        # self attention
        hidden_states =  torch.matmul(residual, self.block.attention.query_key_value.weight.transpose(1,0)) + self.block.attention.query_key_value.bias
        hidden_states = hidden_states.reshape(-1,32,384) # [512,32,384]
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

        # kvcache
        k = torch.cat((past_key, k), dim=0)
        value = torch.cat((past_value, value), dim=0)
        
        # compute attention score
        attention_scores = torch.matmul(q.transpose(0,1)/coeff, k.transpose(0,1).transpose(1,2)) # [32,512,512]
        attention_scores.masked_fill_(attention_mask, -10000.0)
        attention_scores = attention_scores * self.query_key_layer_scaling_coeff
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1) # [32,512,512]
        
        hidden_states = torch.matmul(attention_probs, value.transpose(0,1)) # [32,512,128]
        hidden_states = hidden_states.transpose(0,1) # [512,32,128]
        hidden_states = hidden_states.reshape(-1,4096) # [512,4096]
        hidden_states = torch.matmul(hidden_states,self.block.attention.dense.weight.transpose(1,0))+self.block.attention.dense.bias
        hidden_states = residual * self.alpha + hidden_states
        
        # MLP
        residual = self.block.post_attention_layernorm(hidden_states)
        hidden_states = self.block.mlp(residual)
        hidden_states = residual * self.alpha + hidden_states

        return hidden_states, k, value

rotary_seq_len = 512
# compute cos, sin
inv_freq = block.attention.rotary_emb.inv_freq
t = torch.arange(rotary_seq_len)
freqs = torch.einsum('i,j->ij', t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos_cached = emb.cos()[:, None, :]
sin_cached = emb.sin()[:, None, :]
cos_cached, sin_cached = cos_cached[:rotary_seq_len, ...], sin_cached[:rotary_seq_len, ...]

alpha = (2 * config.num_layers) ** 0.5

layer_id = int(sys.argv[1])
query_key_layer_scaling_coeff = float(layer_id + 1)
coeff = (math.sqrt(128) * query_key_layer_scaling_coeff)



# input
seq_len = 1
torch.manual_seed(42)
input_embed = torch.randn((seq_len, 4096))
position_ids = torch.tensor([[1]])
block_position_ids = torch.tensor([[1]])
attention_mask = torch.zeros(1,seq_len,seq_len,dtype=torch.bool)

# past key value
seq_len = int(sys.argv[2])
past_key = torch.randn(seq_len,32,128)
past_value = torch.randn(seq_len,32,128)

model = Model_lite(block, seq_len, cos_cached, sin_cached, layer_id)


#hidden_states_0 = block.forward(input_embed.unsqueeze(1), torch.cat([position_ids.unsqueeze(1),block_position_ids.unsqueeze(1)],dim=1), attention_mask.unsqueeze(0), layer_id, (past_key.unsqueeze(1), past_value.unsqueeze(1)), True)
# import pdb;pdb.set_trace()
#hidden_states = model.forward(input_embed, position_ids, block_position_ids, past_key, past_value, attention_mask)

#print(hidden_states_0[0])
#print(hidden_states)
torch.onnx.export(model, (input_embed, position_ids, block_position_ids, past_key, past_value, attention_mask), 
                  f'../onnx/glm_kvcache_{sys.argv[1]}.onnx', verbose=False, input_names=['input_embed'], output_names=['hidden_states'])

