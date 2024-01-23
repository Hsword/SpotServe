WORK_DIR=$(pwd)
BASE_DIR="/home/pkuhetu/jfduan/spot_inference"

export PYTHONPATH=$WORK_DIR:$PYTHONPATH

export MASTER_IP=172.24.245.114
export MASTER_PORT=10024
export API_SERVER_PORT=14041
export PARAM_CLIENT_EXEC=${BASE_DIR}/auto-switch-param-client/build/bin/param_client
export FT_INFER_EXEC=${BASE_DIR}/ft-auto-switch/build/bin/multi_gpu_gpt_example_iter


# approach \in [baseline, naive]
APPROACH="naive"
model=$1
TPT=$2
bs=$3
nnodes=$4
strategy=${FIXED_STRATEGY}

CFG=${BASE_DIR}/ft-auto-switch/examples/cpp/multi_gpu_gpt/gpt_config_${model}_bs${bs}.ini
QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/lat_test.txt
CKPT_PATH=${BASE_DIR}/ft-auto-switch/models/megatron-models/c-model/${model}/1-gpu/

export LOG_PATH=$WORK_DIR/log/lat_test/${model}/lat_tpt${TPT}_st${strategy}
mkdir -p $LOG_PATH

echo "Log:" $LOG_PATH

# command
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2
export NCCL_DEBUG=WARN

python $WORK_DIR/main.py \
    --trace-file $WORK_DIR/trace/test_tmp.txt \
    --hostfile $WORK_DIR/trace/hostfile-hetu \
    --query-trace ${QUERY_TRACE} \
    --mbs ${bs} \
    --model-cfg ${CFG} \
    --ckpt-path ${CKPT_PATH} \
    --query-file ${QUERY_FILE} \
    --approach ${APPROACH} \
    --init-pp-deg 1 \
    --nnodes ${nnodes}
