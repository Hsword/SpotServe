#!/bin/bash
source ./scripts_ae/env.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <ablation_level> <trace_name>"
    exit 1
fi

if [ $1 -eq 0 ]; then
    echo "This is equivalent to running \`aws_e2e.sh spotserve 20B $2\`"
    exit 1
elif [ $1 -eq 1 ]; then
    AMODE=overlap
elif [ $1 -eq 2 ]; then
    AMODE=cache
elif [ $1 -eq 3 ]; then
    AMODE=match
elif [ $1 -eq 4 ]; then
    AMODE=plain
else
    echo "Invalid ablation_level: $1, should be one of [0, 1, 2, 3, 4]"
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
MIN_WORLD_SIZE=16
GPU_PER_NODE=4
PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json

if [ $2 == "A" ]; then
    TRACE_NAME="trace_0304_real"
    export LOG_PATH="$WORK_DIR/log/${model}/ablation/${APPROACH}_tpt${TPT}_cv${CV}-0304-$AMODE"
elif [ $2 == "B" ]; then
    TRACE_NAME="trace_0506_real"
    export LOG_PATH="$WORK_DIR/log/${model}/ablation/${APPROACH}_tpt${TPT}_cv${CV}-0506-$AMODE"
else
    echo "Invalid trace: $2, should be one of [A, B]"
    exit 1
fi

mkdir -p $LOG_PATH

python $WORK_DIR/main.py --old-batching \
    --trace-file $WORK_DIR/trace/trace_seg/$TRACE_NAME.txt \
    --hostfile $WORK_DIR/trace/hostfile_aws_T4 \
    --query-trace ${QUERY_TRACE} \
    --mbs ${MAX_BS} \
    --model-cfg ${CFG} \
    --ckpt-path ${CKPT_PATH} \
    --profile-path ${PROFILE_PATH} \
    --query-file ${QUERY_FILE} \
    --approach ${APPROACH} \
    --ablation cache \
    --min-world-size ${MIN_WORLD_SIZE} \
    --nnodes ${MAX_NNODES} \
    --required-tpt ${REQUIRED_TPT} \
    --gpu-per-node ${GPU_PER_NODE} \
    2>&1 | tee ${LOG_PATH}/log.log
