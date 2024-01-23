#!/bin/bash
source ./scripts_ae/env.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <num_node>"
    exit 1
fi

if [ $1==3 ]; then
    num_node=3
elif [ $1==4 ]; then
    num_node=4
elif [ $1==6 ]; then
    num_node=6
elif [ $1==8 ]; then
    num_node=8
else
    echo "Invalid num_node: $1, should be one of [3, 4, 6, 8]"
    exit 1
fi

APPROACH="naive"
TPT=0.35
CV=6
REQUIRED_TPT=0.4
model="h6144"
CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/configs/gpt_config_${model}.ini
QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_tpt${TPT}_cv${CV}.txt
CKPT_PATH=${BASE_DIR}/ckpt/${model}/
# CKPT_PATH=s3://spot-checkpoints/ckpt/${model}/
MAX_BS=4
MAX_NNODES=10
MIN_WORLD_SIZE=12
GPU_PER_NODE=4
PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json

export LOG_PATH="$WORK_DIR/log/${model}/ondemand/${APPROACH}_tpt${TPT}_cv${CV}-node$num_node"
mkdir -p $LOG_PATH

# $WORK_DIR/trace/test.txt
python $WORK_DIR/main.py --old-batching \
    --trace-file $WORK_DIR/trace/trace_seg/trace_0506_node$num_node.txt \
    --hostfile $WORK_DIR/trace/hostfile_aws_T4 \
    --query-trace ${QUERY_TRACE} \
    --mbs ${MAX_BS} \
    --model-cfg ${CFG} \
    --ckpt-path ${CKPT_PATH} \
    --profile-path ${PROFILE_PATH} \
    --query-file ${QUERY_FILE} \
    --approach ${APPROACH} \
    --min-world-size ${MIN_WORLD_SIZE} \
    --nnodes ${MAX_NNODES} \
    --required-tpt ${REQUIRED_TPT} \
    --gpu-per-node ${GPU_PER_NODE} \
    2>&1 | tee ${LOG_PATH}/log.log
