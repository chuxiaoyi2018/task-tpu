#### 第一处修改
transformers/models/marian/modeling_marian.py 第465行
```
present_key_value = present_key_value + cross_attn_present_key_value
```

修改为
```
# present_key_value = present_key_value + cross_attn_present_key_value
```

#### 第二处修改
transformers/models/marian/modeling_marian.py 第85行
```
mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
```
修改为
```
mask = torch.full((tgt_len, tgt_len), -10000, device=device)
```

