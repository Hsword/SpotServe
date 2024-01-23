#!/bin/bash
rm SpotServe-Artifact.zip

zip -r SpotServe-Artifact.zip \
    elastic-switch ParamsClient FasterTransformer \
    ckpt/generate_random_gpt_ckpt.py \
    sync_code.py zip_ae.sh \
    REAMDE.md \
    -x '*.git*' \
    -x '*.pyc' \
    -x '*.ipynb_checkpoints*' \
    -x '*/build/*' \
    -x './elastic-switch/log/*' \
    -x './elastic-switch/graphs/*'
