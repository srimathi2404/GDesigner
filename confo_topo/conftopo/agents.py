"""Agent prompts and the per-claim distribution an agent puts on its claims.

For multiple-choice tasks the atomic sub-claims of a message are
c_k = "the answer is option k". The agent commits to a letter with the first
generated token, so the first-token log-probabilities give its distribution
over claims; the rest of the message is the supporting analysis.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from conftopo.data import Task

# Role descriptions follow G-Designer's MMLU prompt set (copied, not imported).
ROLE_DESCRIPTION: Dict[str, str] = {
    "Mathematician": "You are a mathematician who is good at math games, arithmetic "
                     "calculation, and long-term planning.",
    "Historian": "You research and analyze cultural, economic, political, and social "
                 "events in the past, collect data from primary sources and use it to "
                 "develop theories about what happened during various periods of history.",
    "Doctor": "You are a doctor. You are good at medicine, biology and health, and you "
              "consider the patient's age, lifestyle and medical history when reasoning.",
    "Critic": "You are an excellent critic. Please point out potential issues in other "
              "agents' analysis point by point.",
    "Psychologist": "You are a psychologist. You are good at psychology, sociology, and "
                    "philosophy.",
    "Lawyer": "You are good at law, politics, and history.",
    "Economist": "You are good at economics, finance, and business.",
    "Programmer": "You are good at computer science, engineering, and physics.",
    "Normal": "",
}

# Template 0 is the one whose output is actually forwarded as the message.
# Templates 1.. are meaning-preserving paraphrases used only for the
# semantic (paraphrase) nonconformity score.
TEMPLATES: List[Tuple[str, str]] = [
    ("I will ask you a question and 4 answers enumerated as A, B, C and D.\n"
     "Only one answer out of the offered 4 is correct.\n"
     "Using the reasoning from other agents (if any) as additional advice with critical "
     "thinking, give your answer.\n"
     "Your reply must be less than 80 words: your answer and a brief step by step analysis.\n"
     "The first line of your reply must contain only one letter (A, B, C or D).",
     "The task is: "),
    ("You will be shown a multiple-choice question with four candidate answers labelled "
     "A, B, C and D; exactly one is right. Weigh any analysis provided by other agents "
     "critically rather than copying it. Respond with the letter of the correct answer on "
     "the first line by itself, then a short justification (under 80 words).",
     "Here is the question: "),
    ("Below is a four-option question (A-D) with a single correct option. Other agents' "
     "opinions, when present, are advice only and may be wrong. Begin your reply with just "
     "the letter you choose on its own line; follow with a concise explanation of fewer "
     "than 80 words.",
     "Question: "),
    ("Task: pick the one correct choice among A, B, C, D for the question that follows. "
     "Think carefully; treat other agents' outputs, if given, as fallible hints. Output "
     "format: line 1 = a single letter (A/B/C/D); afterwards a brief step-by-step "
     "rationale, max 80 words.",
     "Please answer the following: "),
    ("Answer the multiple-choice question given by the user. There are four options "
     "(A, B, C, D) and only one is correct. You may consult the other agents' responses "
     "if any are shown, but judge independently. Write only the chosen letter on the "
     "first line, and then explain your choice briefly in under 80 words.",
     "Consider this problem: "),
]

PROBE_SYSTEM = ("You are an independent verifier. You are given a multiple-choice question "
                "(options A, B, C, D; exactly one is correct) and the answer another agent "
                "proposed with its reasoning. The other agent may be wrong. Decide for "
                "yourself which option is correct. Reply with a single letter only.")


def _context_block(spatial: Sequence[dict], temporal: Sequence[dict]) -> str:
    """Render incoming messages. Each item: {'agent': int, 'role': str, 'text': str}.

    This is the hook where later sections replace raw text by the conformal
    message set C_ij (pass already-filtered 'text')."""
    out = ""
    if spatial:
        body = "".join(f"Agent {m['agent']}, role is {m['role']}, output is:\n\n {m['text']}\n\n"
                       for m in spatial)
        out += f"At the same time, the outputs of other agents are as follows:\n\n{body}\n\n"
    if temporal:
        body = "".join(f"Agent {m['agent']}, role is {m['role']}, output is:\n\n {m['text']}\n\n"
                       for m in temporal)
        out += f"In the last round of dialogue, the outputs of other agents were:\n\n{body}"
    return out


def agent_messages(role: str, task: Task, spatial: Sequence[dict] = (),
                   temporal: Sequence[dict] = (), template: int = 0) -> List[dict]:
    constraint, intro = TEMPLATES[template]
    system = (ROLE_DESCRIPTION[role] + "\n" + constraint).strip()
    user = f"{intro}{task.render()}\n" + _context_block(spatial, temporal)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def probe_messages(task: Task, role: str, message_text: str) -> List[dict]:
    user = (f"The question is: {task.render()}\n"
            f"Another agent ({role}) answered:\n\n{message_text}\n\n"
            "Which option is correct? Reply with one letter.")
    return [{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": user}]


def claim_logprobs(first_token_logprobs: Dict[str, float], labels: str) -> Tuple[List[float], float]:
    """Log-distribution over the claims {answer is k}, renormalised over the labels.

    Returns (logprobs per label, letter_mass) where letter_mass is the raw
    probability the model put on *any* answer letter as its first token
    (a sanity diagnostic: should be ~1 if the model follows the format).
    A label absent from the top-k gets the smallest top-k logprob, which is an
    upper bound on its true value."""
    floor = min(first_token_logprobs.values())
    raw = []
    for l in labels:
        cands = [lp for tok, lp in first_token_logprobs.items() if tok.strip() == l]
        raw.append(max(cands) if cands else None)
    mass = sum(math.exp(lp) for lp in raw if lp is not None)
    raw = [lp if lp is not None else floor for lp in raw]
    m = max(raw)
    lse = m + math.log(sum(math.exp(lp - m) for lp in raw))
    return [lp - lse for lp in raw], mass


def parse_choice(text: str, labels: str) -> int:
    """Index of the letter the message commits to (-1 if unparseable)."""
    t = text.strip()
    return labels.index(t[0]) if t and t[0] in labels else -1
