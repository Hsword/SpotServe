import numpy as np
import os

output_dir = './20Bdiv4'
os.system('mkdir -p ' + output_dir)

tp = 4
layer_num = 1

max_seq_len = 1024
hidden_units = 6144
head_num = 48
inter_size = hidden_units * 4
vocab_size = 51200

prefix = 'model.layers.'
suffixes = '.input_layernorm.bias.bin', '.input_layernorm.weight.bin', '.attention.query_key_value.weight.0.bin', \
'.attention.query_key_value.bias.0.bin', '.attention.dense.weight.0.bin', '.attention.dense.bias.bin', \
'.post_attention_layernorm.bias.bin', '.post_attention_layernorm.weight.bin', '.mlp.dense_h_to_4h.weight.0.bin', \
'.mlp.dense_h_to_4h.bias.0.bin', '.mlp.dense_4h_to_h.weight.0.bin', '.mlp.dense_4h_to_h.bias.bin'
size_in_layer = [
    (hidden_units,), 
    (hidden_units,), 
    (hidden_units * 3 // tp, hidden_units),
    (3, hidden_units), 
    (hidden_units // tp, hidden_units),
    (hidden_units // tp,), 
    (hidden_units,), 
    (hidden_units,), 
    (hidden_units // tp, inter_size),
    (inter_size // tp,),
    (inter_size // tp, hidden_units),
    (hidden_units,)
]

non_block_name = 'model.wpe.bin', 'model.wte.bin', 'model.final_layernorm.bias.bin', 'model.final_layernorm.weight.bin'
size_non_block = [
    (max_seq_len, hidden_units),
    (vocab_size, hidden_units),
    (hidden_units,),
    (hidden_units,)
]

for shape, name in zip(size_non_block, non_block_name):
    a = np.random.normal(size=shape).astype(np.float32)
    print(f'shape {shape} to file {name}')
    a.tofile(os.path.join(output_dir, name))

# exit(0)

for i in range(layer_num):
    for shape, suffix in zip(size_in_layer, suffixes):
        a = np.random.normal(size=shape).astype(np.float32)
        name = f'{prefix}{i}{suffix}'
        print(f'shape {shape} to file {name}')
        a.tofile(os.path.join(output_dir, name))