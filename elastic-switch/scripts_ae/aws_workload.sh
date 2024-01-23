#!/bin/bash
source ./scripts_ae/env.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <approach> <trace_name>"
    exit 1
fi

if [ $1 == "reparallelization" ]; then
    APPROACH="baseline"
elif [ $1 == "rerouting" ]; then
    APPROACH="baseline-triton"
elif [ $1 == "spotserve" ]; then
    APPROACH="naive"
else
    echo "Invalid approach: $1, should be one of [reparallelization, rerouting, spotserve]"
    exit 1
fi

CV=6
REQUIRED_TPT=0.6
model="h6144"
CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/configs/gpt_config_${model}.ini
QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_realAr_cv${CV}.txt
CKPT_PATH=${BASE_DIR}/../spot_inference/ckpt/${model}/
# CKPT_PATH=s3://spot-checkpoints/ckpt/${model}/
MAX_BS=4
MAX_NNODES=12
MIN_WORLD_SIZE=12
GPU_PER_NODE=4
PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json


# trace_name in [A, B]
if [ $2 == "A" ]; then
    TRACE_NAME="trace_0304_workload"
    export LOG_PATH="$WORK_DIR/log/${model}/workload/${APPROACH}_realAr_cv${CV}-0304"
elif [ $2 == "B" ]; then
    TRACE_NAME="trace_0506_workload"
    export LOG_PATH="$WORK_DIR/log/${model}/workload/${APPROACH}_realAr_cv${CV}-0506"
else
    echo "Invalid trace: $2, should be one of [A, B]"
    exit 1
fi

mkdir -p $LOG_PATH

# $WORK_DIR/trace/test.txt
python $WORK_DIR/main.py \
    --trace-file $WORK_DIR/trace/trace_seg/$TRACE_NAME.txt \
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
