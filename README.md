# SpotServe Artifact
*SpotServe: Serving Generative Large Language Models on Preemptible Instances* [ASPLOS'24] [Paper Link](https://arxiv.org/abs/2311.15566)

This is the artifact for SpotServe, including codes and scripts for reproducing all experiments in the paper. 

**Note:** Detailed usage guides for each components are not included in the artifact branch. It will soon be available in the main branch.

We require twelve network-accessible GPU instances, each with 4 NVIDIA Tesla T4 GPUs (e.g. AWS `g4dn.12xlarge`), all of which require CUDA, NCCL, MPI, Python dependencies to be installed.
This artifact consists of three components: Global Server (i.e. Inference Server), Params Client (i.e. Context Daemon), and modified FasterTransformer (i.e. Inference Engine). The first component is written in Python, while the other two are in C++. Our provided scripts will automatically launch all of them to perform experiment.

## Requirements

### Hardware dependencies
We conduct expertiments on twelve AWS `g4dn.12xlarge` instances, each of them equipped with four NVIDIA Tesla T4 GPUs and `x86_64` CPU. All instances are connected with each other by TCP/IP with 50Gbps bandwidth.

### Software dependencies
Following toolkits are required: CUDA>=10.2, NCCL>=2.10, MPI, and CMake>=3.8 is highly recommended for building the components.

## Installation

To install the artifact, users need to build ParamsClient and our modified FasterTransformer individually. It is recommended that compile the components on single instance and send them to other nodes by `rsync` command later (See Experiment workflow). 

### Install FasterTransformer
If dependencies are not statisfied, CMake will report the missing dependencies:
```sh
cd ./FasterTransformer
mkdir build && cd build
cmake -DSM=75 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON ..
make multi_gpu_gpt_example_iter -j 8
```

### Install ParamsClient:
```sh
cd ./ParamsClient
mkdir build && cd build
cmake ..
make -j 8
```

### Preparing Checkpoints
Since we focus on the end-to-end latency, using randomized checkpoints is acceptable, we provide a python script to randomly generate model checkpoints. To save disk space, the first layer weights are the only generated files, all weights in succeeding layers are linked to the corresponding files of the first layer. Following command will generate checkpoint files for specified model that can be directly used by out system, available candidates of `model_name` are `6.7B, 20B, 30B`.
```sh
cd ./ckpt
python generate_random_gpt_ckpt.py -o <model_name>
```

### Configure Environment
These files are required to be configured:
* `./elastic_switch/trace/hostfile_aws_T4`: The IP address of your instances, one entry each line, and at least 12 entries.
* `./elastic_switch/scripts_ae/env.sh`: Set NIC, path to MPI, and your base directory. See its contents for details.

### Sync Codes and Data
Make sure that all nodes are accessible to each other, and the Hostfile has been configured. We provide a Python script to automatically send built components and checkpoints (optional) to all the instances. Please set base directory and the IP address where components are built in `sync_code.py`, and run following command:
```sh
python sync_code.py  --n 12 --sync-dataset -hostfile ./elastic-switch/trace/hostnameT4
```

## Experiment workflow
Please set working directory to `./elastic-switch`, and follow `./elastic-switch/README.md` to conduct experiments.

```
@article{asplos24spotserve,
  title = {SpotServe: Serving Generative Large Language Models on Preemptible Instances},
  author = {Miao, Xupeng and Shi, Chunan and Duan, Jiangfei and Xi, Xiaoli and Lin, Dahua and Cui, Bin and Jia, Zhihao},
  journal = {Proceedings of ASPLOS Conference},
  eprint={2311.15566},
  archivePrefix={arXiv},
  year = {2024}
}
```
