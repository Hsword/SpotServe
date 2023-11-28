# SpotServe
*SpotServe: Serving Generative Large Language Models on Preemptible Instances* [ASPLOS'24] [Paper Link](https://arxiv.org/abs/2311.15566)

SpotServe is the first distributed LLM serving system on preemptible instances. Several key techniques in SpotServe realize fast and reliable serving of generative LLMs on cheap preemptible instances. First, SpotServe dynamically adapts the LLM parallelization configuration for dynamic instance availability and fluctuating workload, while balancing the trade-off among the overall throughput, inference latency and monetary costs. Second, to minimize the cost of migrating instances for dynamic reparallelization, the task of migrating instances is formulated as a bipartite graph matching problem, which uses the Kuhn-Munkres algorithm to identify an optimal migration plan that minimizes communications. Finally, to take advantage of the grace period offered by modern clouds, we introduce stateful inference recovery, a new inference mechanism that commits inference progress at a much finer granularity and allows SpotServe to cheaply resume inference upon preemption. We evaluate on real spot instance preemption traces and various popular LLMs and show that SpotServe can reduce the P99 tail latency by 2.4 - 9.1x compared with the best existing LLM serving systems. We also show that SpotServe can leverage the price advantage of preemptive instances, saving 54% monetary cost compared with only using on-demand instances.

We are still preparing artifact evalution and will release the code soon!


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
