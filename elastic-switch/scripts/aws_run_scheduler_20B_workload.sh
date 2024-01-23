WORK_DIR=$(pwd)

# export LD_LIBRARY_PATH=/home/ubuntu/spot_infer_copy/aws-s3-sdk/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export NCCL_SOCKET_IFNAME=ens5
export MPI_EXEC=/opt/amazon/openmpi/bin/mpirun

BASE_DIR="/home/ubuntu/spot_inference"

# approach \in [baseline, baseline-triton, naive]
APPROACH="naive" #
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

export MASTER_IP=172.31.6.17
export MASTER_PORT=10224
export API_SERVER_PORT=14141
export PARAM_CLIENT_EXEC=${BASE_DIR}/ParamsClient/build/bin/param_client
export FT_INFER_EXEC=${BASE_DIR}/FasterTransformer/build/bin/multi_gpu_gpt_example_iter

export LOG_PATH="$WORK_DIR/log/${model}/${APPROACH}_realAr_cv${CV}-0506-nobsz"
mkdir -p $LOG_PATH

# $WORK_DIR/trace/test.txt
python $WORK_DIR/main.py \
    --trace-file $WORK_DIR/trace/trace_seg/trace_0506_workload.txt \
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
