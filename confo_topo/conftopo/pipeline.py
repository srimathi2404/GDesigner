"""Run agents over a topology and log every message as a record.

One JSON line per task. Each node entry (round t, agent i) stores the raw
message m_i^t and the *signals* that nonconformity scores are computed from:

    lp_self   log p_i(claim k | context)          agent's own first-token distribution
    lp_para   same, under paraphrased prompts     (semantic / paraphrase score)
    lp_probe  probe agent's distribution after reading m_i^t  (disagreement score)

Scores themselves are computed offline (scores.py), so new scores that reuse
these signals need no new LLM calls. Collection is resumable.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List

from tqdm import tqdm

from conftopo.agents import (TEMPLATES, agent_messages, claim_logprobs, parse_choice,
                             probe_messages)
from conftopo.data import CATEGORY_FN, Task
from conftopo.llm import LLM
from conftopo.topology import TOPOLOGIES, predecessors, topological_order


@dataclass
class PipelineConfig:
    roles: List[str] = field(default_factory=lambda: ["Mathematician", "Historian", "Doctor", "Critic"])
    topology: str = "mesh"
    rounds: int = 2
    max_tokens: int = 160
    n_paraphrases: int = 4   # extra prompt templates for the paraphrase score (0 disables)
    probe: bool = True       # collect the probe-agent signal

    def __post_init__(self):
        assert self.n_paraphrases <= len(TEMPLATES) - 1


async def run_task(task: Task, cfg: PipelineConfig, llm: LLM) -> dict:
    n = len(cfg.roles)
    edges = TOPOLOGIES[cfg.topology](n)
    preds = predecessors(edges, n)
    order = topological_order(edges, n)
    labels = task.labels

    nodes: List[dict] = []
    side_calls = []  # (node, key, awaitable) - signals that don't feed later agents
    prev_round: List[dict] = []
    for r in range(cfg.rounds):
        this_round: dict = {}
        for i in order:
            role = cfg.roles[i]
            spatial = [this_round[p] for p in preds[i]]
            temporal = prev_round
            res = await llm.chat(agent_messages(role, task, spatial, temporal), cfg.max_tokens)
            lp, mass = claim_logprobs(res.first_token_logprobs, labels)
            node = {"round": r, "agent": i, "role": role, "preds": preds[i],
                    "text": res.text, "choice": parse_choice(res.text, labels),
                    "lp_self": lp, "letter_mass": mass,
                    "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}
            nodes.append(node)
            this_round[i] = {"agent": i, "role": role, "text": res.text}
            for t in range(1, cfg.n_paraphrases + 1):
                side_calls.append((node, "lp_para", asyncio.ensure_future(
                    llm.chat(agent_messages(role, task, spatial, temporal, template=t), 1))))
            if cfg.probe:
                side_calls.append((node, "lp_probe", asyncio.ensure_future(
                    llm.chat(probe_messages(task, role, res.text), 1))))
        prev_round = [this_round[i] for i in range(n)]

    results = await asyncio.gather(*(c for _, _, c in side_calls))
    for (node, key, _), res in zip(side_calls, results):
        lp, _ = claim_logprobs(res.first_token_logprobs, labels)
        if key == "lp_para":
            node.setdefault("lp_para", []).append(lp)
        else:
            node[key] = lp

    return {"task_id": task.task_id, "dataset": task.dataset, "split": task.split,
            "subject": task.subject,
            "category": CATEGORY_FN.get(task.dataset, lambda s: "all")(task.subject),
            "answer": task.answer, "labels": labels,
            "edges": [list(e) for e in edges], "nodes": nodes}


async def collect(tasks: List[Task], cfg: PipelineConfig, llm: LLM, out_dir: str,
                  task_concurrency: int = 24) -> str:
    """Run the pipeline on every task not already in out_dir/records.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "records.jsonl")
    cfg_path = os.path.join(out_dir, "config.json")
    run_cfg = {"pipeline": asdict(cfg), "llm": asdict(llm.cfg)}
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if old["pipeline"] != run_cfg["pipeline"] or old["llm"]["model"] != run_cfg["llm"]["model"]:
            raise ValueError(f"{out_dir} was collected with a different config; use a new --run name")
    else:
        json.dump(run_cfg, open(cfg_path, "w"), indent=2)

    done = set()
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            done = {json.loads(line)["task_id"] for line in f if line.strip()}
    todo = [t for t in tasks if t.task_id not in done]
    print(f"{len(done)} tasks already collected, {len(todo)} to run -> {rec_path}")

    sem = asyncio.Semaphore(task_concurrency)
    failed = 0

    async def worker(task: Task):
        async with sem:
            try:
                return await run_task(task, cfg, llm)
            except Exception as e:
                print(f"[skip] {task.task_id}: {e!r}")
                return None

    with open(rec_path, "a") as f:
        futs = [asyncio.ensure_future(worker(t)) for t in todo]
        for fut in tqdm(asyncio.as_completed(futs), total=len(futs), mininterval=10):
            rec = await fut
            if rec is None:
                failed += 1
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
    u = llm.usage
    print(f"done. failed={failed} calls={u.calls} prompt_tokens={u.prompt_tokens} "
          f"completion_tokens={u.completion_tokens}")
    return rec_path


def load_records(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
