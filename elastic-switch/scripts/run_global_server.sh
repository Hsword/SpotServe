#!/bin/sh
WORK_DIR=$(pwd)
# echo $WORK_DIR
PYTHONPATH=$PYTHONPATH:$WORK_DIR python $WORK_DIR/global_server/main.py $*
