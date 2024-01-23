import os
import io
import boto3
import time
import torch
import numpy as np

prefix = 'model.layers.'
suffixes = '.input_layernorm.bias.bin', '.input_layernorm.weight.bin', '.attention.query_key_value.weight.0.bin', \
'.attention.query_key_value.bias.0.bin', '.attention.dense.weight.0.bin', '.attention.dense.bias.bin', \
'.post_attention_layernorm.bias.bin', '.post_attention_layernorm.weight.bin', '.mlp.dense_h_to_4h.weight.0.bin', \
'.mlp.dense_h_to_4h.bias.0.bin', '.mlp.dense_4h_to_h.weight.0.bin', '.mlp.dense_4h_to_h.bias.bin'


def cli_cp(filename):
    os.system('mkdir -p test_ckpt')
    cmd = f'aws s3 cp s3://spot-checkpoints/ckpt/20Bdiv4/{filename} test_ckpt/{filename}'
    st = time.time()
    os.system(cmd)
    ed = time.time()
    os.system('rm -rf test_ckpt')
    print(f'cost {ed - st}s')


def direct_read(filename):
    # os.system('mkdir -p test_ckpt')
    s3 = boto3.client('s3')
    st = time.time()
    with io.BytesIO() as f:
        print('loading', filename)
        s3.download_fileobj('spot-checkpoints', f'ckpt/20Bdiv4/{filename}', f)
        f.seek(0)
        a = torch.Tensor(np.frombuffer(f.read()), deivce='cuda:0')
    ed = time.time()
    print(filename, a.shape, f'cost {ed - st}s')
    # os.system('rm -rf test_ckpt')
    del a


if __name__ == '__main__':
    assert torch.cuda.is_available()
    
    name = ['model.wpe.bin', 'model.wte.bin', 'model.final_layernorm.bias.bin', 'model.final_layernorm.weight.bin']
    
    for suffix in suffixes:
        name.append(f'{prefix}0{suffix}')
    
    for n in name:
        direct_read(name)
    # cli_cp('model.layers.0.mlp.dense_4h_to_h.weight.0.bin')
