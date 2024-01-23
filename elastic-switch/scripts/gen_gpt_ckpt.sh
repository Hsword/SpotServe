FT_DIR="/home/duanjiangfei/spot_inference/ft-auto-switch/"

NPARAM=${1:-345m}
layer_num=${2:-32}
hidden_units=${3:-4096}
max_seq_len=${4:-1024}
vocab_size=${5:-51200}

MODEL_DIR=${FT_DIR}/models/megatron-models/c-model/${NPARAM}/1-gpu

mkdir -p ${MODEL_DIR}

python scripts/generate_gpt_ckpt.py --output-dir ${MODEL_DIR} \
    --n ${layer_num} \
    --h ${hidden_units} \
    --seq ${max_seq_len} \
    --vocab ${vocab_size}
