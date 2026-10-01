"""Tasks and dataset loaders.

A Task is a question with a finite set of candidate claims ("the answer is
option k") and one ground-truth claim. To add a dataset, write a loader that
returns List[Task] and register it in DATASETS.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

LABELS = "ABCDEFGH"

CONFTOPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Read-only: the MMLU CSVs already downloaded for the G-Designer smoke test.
DEFAULT_MMLU_ROOT = os.path.abspath(
    os.path.join(CONFTOPO_ROOT, "..", "GDesigner", "datasets", "MMLU", "data"))


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    split: str
    subject: str
    question: str
    options: Tuple[str, ...]
    answer: int  # index into options

    @property
    def labels(self) -> str:
        return LABELS[:len(self.options)]

    def render(self) -> str:
        """Same textual format G-Designer feeds its MMLU agents."""
        lines = [str(self.question)]
        lines += [f"Option {l}: {o}" for l, o in zip(self.labels, self.options)]
        return "\n".join(lines) + "\n"


# Subject -> coarse category, from the official MMLU categories.py.
_MMLU_SUBCATEGORY = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics",
    "business_ethics": "business", "clinical_knowledge": "health",
    "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math",
    "college_medicine": "health", "college_physics": "physics",
    "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering",
    "elementary_mathematics": "math", "formal_logic": "philosophy",
    "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics", "high_school_mathematics": "math",
    "high_school_microeconomics": "economics", "high_school_physics": "physics",
    "high_school_psychology": "psychology", "high_school_statistics": "math",
    "high_school_us_history": "history", "high_school_world_history": "history",
    "human_aging": "health", "human_sexuality": "culture",
    "international_law": "law", "jurisprudence": "law",
    "logical_fallacies": "philosophy", "machine_learning": "computer science",
    "management": "business", "marketing": "business", "medical_genetics": "health",
    "miscellaneous": "other", "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy", "nutrition": "health", "philosophy": "philosophy",
    "prehistory": "history", "professional_accounting": "other",
    "professional_law": "law", "professional_medicine": "health",
    "professional_psychology": "psychology", "public_relations": "politics",
    "security_studies": "politics", "sociology": "culture",
    "us_foreign_policy": "politics", "virology": "health",
    "world_religions": "philosophy",
}
_MMLU_CATEGORY = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}
_SUB2CAT = {sub: cat for cat, subs in _MMLU_CATEGORY.items() for sub in subs}


def mmlu_category(subject: str) -> str:
    return _SUB2CAT.get(_MMLU_SUBCATEGORY.get(subject, "other"), "other")


def load_mmlu(split: str, root: str = DEFAULT_MMLU_ROOT) -> List[Task]:
    tasks: List[Task] = []
    paths = sorted(glob.glob(os.path.join(root, split, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No MMLU csvs under {os.path.join(root, split)}")
    for path in paths:
        subject = os.path.basename(path)[:-len(f"_{split}.csv")]
        df = pd.read_csv(path, header=None, encoding="utf-8",
                         names=["question", "A", "B", "C", "D", "answer"],
                         dtype=str, keep_default_na=False)
        for i, row in df.iterrows():
            tasks.append(Task(
                task_id=f"mmlu/{split}/{subject}/{i}", dataset="mmlu", split=split,
                subject=subject, question=row["question"],
                options=(row["A"], row["B"], row["C"], row["D"]),
                answer="ABCD".index(row["answer"].strip())))
    return tasks


DATASETS: Dict[str, Callable[..., List[Task]]] = {"mmlu": load_mmlu}
CATEGORY_FN: Dict[str, Callable[[str], str]] = {"mmlu": mmlu_category}


def sample_tasks(tasks: List[Task], n: int | None, seed: int) -> List[Task]:
    """Uniform random subsample without replacement (keeps the subject mix)."""
    if n is None or n >= len(tasks):
        return list(tasks)
    idx = np.random.default_rng(seed).choice(len(tasks), size=n, replace=False)
    return [tasks[i] for i in sorted(idx)]
