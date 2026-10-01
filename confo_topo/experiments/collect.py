"""Collect agent messages + score signals on labelled tasks.

    python experiments/collect.py --run mmlu_mesh4 --n_test 1500

Calibration data = labelled tasks from the same distribution as the tasks you
will deploy on, pushed through the *same* agent pipeline. Here: all of MMLU
`val` plus a uniform random sample of MMLU `test`.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conftopo.data import CONFTOPO_ROOT, DATASETS, sample_tasks
from conftopo.llm import LLM, LLMConfig
from conftopo.pipeline import PipelineConfig, collect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mmlu_mesh4", help="name of the run directory under runs/")
    p.add_argument("--dataset", default="mmlu")
    p.add_argument("--n_val", type=int, default=None, help="tasks from val (default: all)")
    p.add_argument("--n_test", type=int, default=1500, help="tasks sampled from test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roles", nargs="+", default=["Mathematician", "Historian", "Doctor", "Critic"])
    p.add_argument("--topology", default="mesh")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--n_paraphrases", type=int, default=4)
    p.add_argument("--no_probe", action="store_true")
    p.add_argument("--task_concurrency", type=int, default=24)
    p.add_argument("--max_concurrency", type=int, default=64)
    a = p.parse_args()

    load = DATASETS[a.dataset]
    tasks = sample_tasks(load("val"), a.n_val, a.seed) + sample_tasks(load("test"), a.n_test, a.seed)
    cfg = PipelineConfig(roles=a.roles, topology=a.topology, rounds=a.rounds,
                         n_paraphrases=a.n_paraphrases, probe=not a.no_probe)
    llm = LLM(LLMConfig(max_concurrency=a.max_concurrency))
    out_dir = os.path.join(CONFTOPO_ROOT, "runs", a.run)
    asyncio.run(collect(tasks, cfg, llm, out_dir, a.task_concurrency))


if __name__ == "__main__":
    main()
