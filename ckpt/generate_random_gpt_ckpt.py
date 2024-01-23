import numpy as np
import os
import argparse

def gen_ckpt(output_dir):
    os.system('mkdir -p ' + output_dir)

    tp = 1
    max_seq_len = 1024
    vocab_size = 51200
    
    if output_dir == '6.7B':
        layer_num = 36 # real_layer is 32, 4 of them are considered as empty layers by FT
        hidden_units = 4096
    elif output_dir == '20B':
        output_dir = 'h6144'
        layer_num = 48 # real_layer is 44, 4 of them are considered as empty layers by FT
        hidden_units = 6144
    elif output_dir == '30B':
        output_dir = 'h7168'
        layer_num = 48
        hidden_units = 7168
    else:
        raise NotImplementedError
        
    inter_size = hidden_units * 4

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

    for i in range(layer_num):
        for shape, suffix in zip(size_in_layer, suffixes):
            name = f'{prefix}{i}{suffix}'
            if i == 0:
                a = np.random.normal(size=shape).astype(np.float32)
                print(f'shape {shape} to file {name}')
                a.tofile(os.path.join(output_dir, name))
            elif not os.path.exists(os.path.join(output_dir, name)):
                os.link(os.path.join(output_dir, f'{prefix}0{suffix}'), os.path.join(output_dir, name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, choices=['6.7B', '20B', '30B'], required=True)
    args = parser.parse_args()
    gen_ckpt(args.output_dir)
    print('Done.')