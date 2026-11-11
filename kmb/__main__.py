"""Command line: ``python -m kmb {run,data,plot,list}``."""

from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m kmb", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the fits of a config, appending to results/<name>.jsonl")
    p.add_argument("config")
    p.add_argument("--machine", help="machine label (default: config, $KMB_MACHINE, hostname)")
    p.add_argument("--impl", action="append", help="only these implementations (repeatable)")
    p.add_argument("--dataset", action="append", help="only these datasets (repeatable)")
    p.add_argument("--k", action="append", type=int, help="only these k (repeatable)")
    p.add_argument("--threads", type=int, help="override the config's thread count")
    p.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    p.add_argument("--max-rounds", type=int,
                   help="run only the first N rounds (one job per implementation each) of "
                        "every comparison: a quick way to estimate the duration of a sweep")
    p.add_argument("--retry-errors", action="store_true", help="rerun fits that failed")

    p = sub.add_parser("data", help="download and convert datasets to float32 .npy")
    p.add_argument("names", nargs="*", help="dataset names (default: list them)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("plot", help="one Pareto-front figure per (dataset, k, machine, device)")
    p.add_argument("results", nargs="*", help="JSONL files (default: results/*.jsonl)")
    p.add_argument("--out", default="plots")
    p.add_argument("--split-by", help="one front per value of this setting (e.g. init)")

    sub.add_parser("list", help="list implementations and their default settings")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        from kmb import config, runner

        cfg = config.load(args.config, machine=args.machine)
        if args.threads:
            cfg.threads = args.threads
        if args.dataset or args.k:
            cfg.datasets = [(d, [k for k in ks if not args.k or k in args.k])
                            for d, ks in cfg.datasets if not args.dataset or d.name in args.dataset]
        runner.run(cfg, impl_filter=args.impl, dry_run=args.dry_run, max_rounds=args.max_rounds,
                   retry_errors=args.retry_errors)
    elif args.cmd == "data":
        from kmb import datasets

        if not args.names:
            for d in datasets.DATASETS.values():
                state = "cached" if datasets.path(d.name).exists() else "missing"
                print(f"{d.name:10s} {str(d.shape):18s} {state:8s} {d.source}")
        for name in args.names:
            datasets.prepare(name, force=args.force)
    elif args.cmd == "plot":
        from kmb import plot

        plot.main(args.results or None, args.out, args.split_by)
    elif args.cmd == "list":
        from kmb import impls

        for name, (env, device, _, _) in impls.REGISTRY.items():
            print(f"{name:18s} env={env:10s} {device}  {impls.defaults_of(name)}")


if __name__ == "__main__":
    main()
