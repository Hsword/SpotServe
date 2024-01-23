WORK_DIR=$(pwd)

export PYTHONPATH=$WORK_DIR:$PYTHONPATH

BASE_DIR="/home/duanjiangfei/spot_inference"

# approach \in [baseline, naive]
APPROACH="naive"
TPT=4
model="345M"
CFG=${BASE_DIR}/ft-auto-switch/examples/cpp/multi_gpu_gpt/gpt_config.ini
QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_tpt${TPT}.txt
CKPT_PATH=${BASE_DIR}/ft-auto-switch/models/megatron-models/c-model/${model}/1-gpu/
MAX_BS=4
MAX_NNODES=8
MIN_WORLD_SIZE=2
GPU_PER_NODE=1
PROFILE_PATH=${BASE_DIR}/elastic-switch/profile/T4-4x/megatron_${model}_profile.json

export MASTER_IP=10.1.72.217
export MASTER_PORT=10024
export API_SERVER_PORT=14041
export PARAM_CLIENT_EXEC=${BASE_DIR}/auto-switch-param-client/build/bin/param_client
export FT_INFER_EXEC=${BASE_DIR}/ft-auto-switch/build/bin/multi_gpu_gpt_example_iter

export LOG_PATH=$WORK_DIR/log/${model}/${APPROACH}_tpt${TPT}
mkdir -p $LOG_PATH

python $WORK_DIR/main.py \
    --trace-file $WORK_DIR/trace/test.txt \
    --hostfile $WORK_DIR/trace/hostfile \
    --query-trace ${QUERY_TRACE} \
    --mbs ${MAX_BS} \
    --model-cfg ${CFG} \
    --ckpt-path ${CKPT_PATH} \
    --query-file ${QUERY_FILE} \
    --profile-path ${PROFILE_PATH} \
    --approach ${APPROACH} \
    --min-world-size ${MIN_WORLD_SIZE} \
    --nnodes ${MAX_NNODES} \
    --gpu-per-node ${GPU_PER_NODE} \
    2>&1 | tee ${LOG_PATH}/log.log
