import argparse

from scheduler.trace_replayer import TraceReplayer
from scheduler.api_server import APIServer
from scheduler.scheduler import Scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    # trace replay args
    parser.add_argument('--trace-file', type=str)
    parser.add_argument('--hostfile', type=str)
    parser.add_argument('--nnodes', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--grace-period', type=int, default=30)
    parser.add_argument('--gpu-per-node', type=int, default=1)
    # approach args
    parser.add_argument('--approach', type=str, default=None)
    # ablation: ['plain', 'match', 'cache', 'overlap]
    parser.add_argument('--ablation', type=str, default=None)
    # inference job args
    parser.add_argument('--query-trace', type=str)
    parser.add_argument('--query-file', type=str)
    parser.add_argument('--required-tpt', type=float, default=None)
    parser.add_argument('--mbs', type=int, default=1)
    parser.add_argument('--init-strategy', nargs='+', type=int, default=None)
    parser.add_argument('--min-world-size', type=int, default=1)
    parser.add_argument('--old-batching', action='store_true')
    # model args
    parser.add_argument('--model-cfg', type=str)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--profile-path', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = args.model_cfg
    trace_replayer = TraceReplayer(args.trace_file, args.hostfile, args.nnodes, args.gpu_per_node, args.dry_run)
    checkpoint_path = args.ckpt_path
    profile_path = args.profile_path

    query_trace = args.query_trace
    query_file = args.query_file
    mbs = args.mbs
    init_strategy = args.init_strategy

    scheduler = Scheduler(model_cfg, trace_replayer, checkpoint_path, profile_path, query_trace, query_file, mbs,
                          args.approach, args.required_tpt, init_strategy, args.min_world_size, args.grace_period,
                          args.ablation if args.ablation != 'naive' else None, not args.old_batching)

    # process requests
    scheduler.run()


if __name__ == '__main__':
    main()
