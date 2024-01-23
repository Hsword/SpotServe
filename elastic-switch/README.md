# SpotServe Artifact

**Note:** Please complete the steps in `../README.md` first. We assume everything has been prepared for conducting experiments.

## Experiment workflow

### Performing Experiments
We provide shell scripts to generate per-request end-to-end latency, which will be used for plotting figures later. All scripts are located in './scripts_ae/', please set working directory to **HERE** before running the following scripts. It is not necessary to run all of them before go to next step.

* `aws_e2e.sh` will start the end-to-end latency evaluation in §6.2. In the following command, the `approach` can be one of `reparallelization, rerouting, spotserve`, and the `model_name` should be one of `6.7B, 20B, 30B`, while the `trace_name` should be one of `As, Bs, As+o, Bs+o`. Each execution will be corresponding to a single curve in Figure 6:
```sh
./scripts_ae/aws_e2e.sh <approach> <model_name> <trace_name>
```
* `aws_ondemand.sh` will start the monetary cost evaluation in §6.2, generating the dashed blue line in Figure 7. The `num_node` can be one of `3, 4, 6, 8`, but the dashed line will be plotted only when all of the four experiments have been conducted:
```sh
./scripts_ae/aws_ondemand.sh <num_node>
```
* `aws_workload.sh` will start the fluctuating workload evaluation in \S6.3 on specified trace (can be either `A` or `B`), where the `approach` can also be one of `reparallelization, rerouting, spotserve`:
```sh
./scripts_ae/aws_workload.sh <approach> <trace_name>
```
* `aws_ablation.sh` will start the ablation study evaluation in \S6.2 on specified trace (can be either `A` or `B`), where the `ablation_level` is from 0 to 4, corresponding the five columns in Figure 9:
```sh
./scripts_ae/aws_ablation.sh <ablation_level> <trace_name>
```

### Plotting Figures
All the scripts above only generate the end-to-end latency for each request. To analysis these data and plot figures presented in the paper, we also provide a `plot.py` together with the scripts:
```sh
python ./scripts_ae/plot.py <mode> [-m MODEL] [-t TRACE]
```
This script works even when only part of the experiment has been completed (just ignoring missing results), allowing users to check partial experimental data.
Here is the available options for `mode`:
* `e2e` - Plot the corresponding figure as in Figure 6, both `-m, -t` flags are required to be specified.
* `price` - Plot the monetary cost comparison figure as Figure 7, in which the scatters come from `aws_e2e.sh`.
* `workload-e2e` - Plot the end-to-end latency figure as Figure 8e, 8f on the trace specified by `-t` flag.
* `workload-case` - Plot the per-request latency figure as Figure 8g, 8h on the trace specified by `-t` flag.
* `ablation` - Plot the ablation study figure as Figure 9.

## Evaluation and expected results
The specific results differ on the hardware, bandwidth, and sometimes sensitive to unpredictable GPU/network/batching fluctuations. However, we expect the results users reproduced roughly match the trends as the figures presented in the paper within the same environment. (i.e. Figure 6, 7, 8, 9)

## Experiment customization
Currently, we do not have a convenient configuration script for custom experiments. The customization guide will be available together with main branch.

## Notes
Occasionally, some processes on certain nodes may not exit even though the evaluation is finished. We provide `kill_all.sh`, and running following command to kill all concerning processes after each experiment is highly recommended:
```sh
./scripts_ae/kill_all.sh 12 ./trace/hostfile_aws_T4
```
