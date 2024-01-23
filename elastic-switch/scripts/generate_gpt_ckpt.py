import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, default='../models/megatron-models/c-model/6.7b/1-gpu')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--h', type=int, default=4096)
parser.add_argument('--vocab', type=int, default=51200)
parser.add_argument('--seq', type=int, default=1024)
args = parser.parse_args()

output_dir = args.output_dir
os.system('mkdir -p ' + output_dir)


layer_num = args.n
max_seq_len = args.seq
hidden_units = args.h
vocab_size = args.vocab

head_num = 32
inter_size = hidden_units * 4

prefix = 'model.layers.'
suffixes = '.input_layernorm.bias.bin', '.input_layernorm.weight.bin', '.attention.query_key_value.weight.0.bin', \
'.attention.query_key_value.bias.0.bin', '.attention.dense.weight.0.bin', '.attention.dense.bias.bin', \
'.post_attention_layernorm.bias.bin', '.post_attention_layernorm.weight.bin', '.mlp.dense_h_to_4h.weight.0.bin', \
'.mlp.dense_h_to_4h.bias.0.bin', '.mlp.dense_4h_to_h.weight.0.bin', '.mlp.dense_4h_to_h.bias.bin'
size_in_layer = [
    (hidden_units,),
    (hidden_units,),
    (hidden_units * 3, hidden_units),
    (3, hidden_units),
    (hidden_units, hidden_units),
    (hidden_units,),
    (hidden_units,),
    (hidden_units,),
    (hidden_units, inter_size),
    (inter_size,),
    (inter_size, hidden_units),
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


for i in range(layer_num):
    for shape, suffix in zip(size_in_layer, suffixes):
        a = np.random.normal(size=shape).astype(np.float32)
        name = f'{prefix}{i}{suffix}'
        print(f'shape {shape} to file {name}')
        a.tofile(os.path.join(output_dir, name))
