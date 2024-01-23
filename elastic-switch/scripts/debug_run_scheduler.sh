WORK_DIR=$(pwd)

export PYTHONPATH=$WORK_DIR:$PYTHONPATH
# export NCCL_SOCKET_IFNAME=ens3
export MPI_EXEC="mpirun --allow-run-as-root"

BASE_DIR="/home/sj/sca/"

# approach \in [baseline, naive]
APPROACH="naive"
TPT=4
model="345M"
CFG=${BASE_DIR}/FasterTransformer/examples/cpp/multi_gpu_gpt/gpt_config.ini
QUERY_FILE=${BASE_DIR}/elastic-switch/trace/query/query_seq512.csv
QUERY_TRACE=${BASE_DIR}/elastic-switch/trace/query/query_tpt${TPT}.txt
CKPT_PATH=${BASE_DIR}/FasterTransformer/models/megatron-models/c-model/345m/1-gpu/
MAX_NNODES=1
GPU_PER_NODE=1

export MASTER_IP=127.0.0.1
export MASTER_PORT=10224
export API_SERVER_PORT=14141
export PARAM_CLIENT_EXEC=${BASE_DIR}/ParamsClient/build/bin/param_client
export FT_INFER_EXEC=${BASE_DIR}/FasterTransformer/build/bin/multi_gpu_gpt_example_iter

export LOG_PATH=$WORK_DIR/log/${model}/${APPROACH}_tpt${TPT}
mkdir -p $LOG_PATH

python $WORK_DIR/main.py \
    --trace-file $WORK_DIR/trace/test.txt \
    --hostfile $WORK_DIR/trace/hostfile_local \
    --query-trace ${QUERY_TRACE} \
    --mbs 1 \
    --model-cfg ${CFG} \
    --ckpt-path ${CKPT_PATH} \
    --query-file ${QUERY_FILE} \
    --approach ${APPROACH} \
    --init-pp-deg 1 \
    --nnodes ${MAX_NNODES} \
    --gpu-per-node ${GPU_PER_NODE}
