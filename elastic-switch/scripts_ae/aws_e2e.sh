#!/bin/bash
source ./scripts_ae/env.sh

FLAGS="--old-batching"
# FLAGS=""

if [ $# -ne 3 ]; then
    echo "Usage: $0 <approach> <model_name> <trace_name>"
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

if [ $2 == "6.7B" ]; then
    model="6.7B"
    TPT=1.5
    CV=6
    REQUIRED_TPT=2
    CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/configs/gpt_config_${model}.ini
    QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query6144_seq512.csv
    QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query6144_tpt${TPT}_cv${CV}.txt
    CKPT_PATH=${BASE_DIR}/ckpt/${model}/
    MAX_BS=4
    MAX_NNODES=10
    MIN_WORLD_SIZE=4
    GPU_PER_NODE=4
    PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json
elif [ $2 == "20B" ]; then
    TPT=0.35
    CV=6
    REQUIRED_TPT=0.4
    model="h6144"
    CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/configs/gpt_config_${model}.ini
    QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
    QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_tpt${TPT}_cv${CV}.txt
    CKPT_PATH=${BASE_DIR}/ckpt/${model}/
    if [ $3 == "As+o" ]; then
        FLAGS=""
    fi
    MAX_BS=4
    MAX_NNODES=10
    MIN_WORLD_SIZE=12
    GPU_PER_NODE=4
    PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json
elif [ $2 == "30B" ]; then
    TPT=0.2
    CV=6
    REQUIRED_TPT=0.2
    model="h7168"
    CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/configs/gpt_config_${model}.ini
    QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
    QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_tpt${TPT}_cv${CV}.txt
    CKPT_PATH=${BASE_DIR}/ckpt/${model}/
    MAX_BS=4
    MAX_NNODES=10
    MIN_WORLD_SIZE=16
    GPU_PER_NODE=4
    PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json
else
    echo "Invalid model: $2, should be one of [6.7B, 20B, 30B]"
    exit 1
fi

# trace_name in [As, Bs, As+o, Bs+o]
if [ $3 == "As" ]; then
    TRACE_NAME="trace_0304_real"
    export LOG_PATH="$WORK_DIR/log/${model}/real/${APPROACH}_tpt${TPT}_cv${CV}-0304"
elif [ $3 == "Bs" ]; then
    TRACE_NAME="trace_0506_real"
    export LOG_PATH="$WORK_DIR/log/${model}/real/${APPROACH}_tpt${TPT}_cv${CV}-0506"
elif [ $3 == "As+o" ]; then
    TRACE_NAME="trace_0304_ondemand"
    export LOG_PATH="$WORK_DIR/log/${model}/ondemand/${APPROACH}_tpt${TPT}_cv${CV}-0304"
elif [ $3 == "Bs+o" ]; then
    TRACE_NAME="trace_0506_ondemand"
    export LOG_PATH="$WORK_DIR/log/${model}/ondemand/${APPROACH}_tpt${TPT}_cv${CV}-0506"
else
    echo "Invalid trace: $3, should be one of [As, Bs, As+o, Bs+o]"
    exit 1
fi

mkdir -p $LOG_PATH

# $WORK_DIR/trace/test.txt
python $WORK_DIR/main.py $FLAGS \
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
