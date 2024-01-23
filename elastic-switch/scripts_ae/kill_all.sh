#!/bin/bash
nnode=$1
ip_file=${2:-"hostname"}
machines=$(cat $ip_file)

for node in $machines
do
    # skip if node is in the format of slots=xx
    case "$node" in
    slots=*) continue ;;
    *)
    esac
    echo "Node: $node"
    ssh -o StrictHostKeyChecking=no ubuntu@${node} "pkill -9 -f multi_gpu_gpt_example_iter"
    ssh -o StrictHostKeyChecking=no ubuntu@${node} "pkill -9 -f param_client"
    ssh -o StrictHostKeyChecking=no ubuntu@${node} "pkill -9 -f main.py"

    nnode=$((nnode-1))
    if [ $nnode -eq 0 ]; then
        break
    fi
done
