"""ConfTopo: conformal multi-agent topologies (empirical checks of the tech report).

Layout (one module per concern, so later sections plug in without rewrites):

    data.py       tasks + dataset loaders            (add MedQA / MATH / ... here)
    llm.py        async OpenAI-compatible client     (logprobs, concurrency, retries)
    agents.py     role prompts, claim distributions  (what an agent says + how sure)
    topology.py   DAG builders                       (4.2 / 4.3 grow these)
    pipeline.py   run agents over a topology, log every message as a record
    scores.py     nonconformity scores s(c | context)            (section 4.1)
    conformal.py  split-CP thresholds, prediction sets, gating   (section 4.1)

Experiments live in ../experiments and only read the records written by pipeline.py.
"""
