WORK_DIR=$(pwd)

PYTHONPATH=$PYTHONPATH:$WORK_DIR python -m torch.distributed.launch --nproc_per_node=8 $WORK_DIR/servers/bert_param_server.py