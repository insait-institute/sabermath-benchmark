"""Command-line interface: `sabermath evaluate | download | help`."""

import argparse
import sys
from typing import get_args

from sabermath.schemas import Task


def _evaluate(args):
    if args.vllm and args.device:
        sys.exit("error: --device is not supported with --vllm")

    from sabermath import evaluate

    report = evaluate(
        args.model,
        tasks=args.task,
        use_vllm=args.vllm,
        init_kwargs={"device": args.device} if args.device else None,
    )

    print(f"\n{report.model} ({report.processor}), nDCG@{report.k}")
    for task in report.tasks:
        print(f"\n{task.task:24}{task.ndcg_at_k:.4f}")
        for branch in task.branches:
            print(f"  {branch.branch:22}{branch.ndcg_at_k:.4f}")


def _download(args):
    from sabermath.data import load_data

    queries, documents = load_data()
    print(f"Downloaded {len(queries)} queries and {len(documents)} documents.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sabermath", description="SABER-Math: a reranking benchmark for math retrieval."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("evaluate", help="Evaluate a model on the benchmark.")
    p.add_argument("model", help="Hugging Face model id.")
    p.add_argument("--task", action="append", choices=get_args(Task),
                   help="Task to run; repeat for several (default: all three).")
    p.add_argument("--device", help="Device for Sentence Transformers, e.g. cuda or cpu.")
    p.add_argument("--vllm", action="store_true", help="Load the model with vLLM.")
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("download", help="Download the benchmark data into the cache.")
    p.set_defaults(func=_download)

    p = sub.add_parser("help", help="Show help for sabermath or one of its commands.")
    p.add_argument("topic", nargs="?", choices=["evaluate", "download", "help"])

    args = parser.parse_args(argv)
    if args.command in (None, "help"):
        topic = getattr(args, "topic", None)
        (sub.choices[topic] if topic else parser).print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
