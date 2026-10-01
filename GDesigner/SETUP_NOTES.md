# Setup status (against instructions.md)

Everything is done except actually starting the vLLM server and running the
smoke test — both need a free GPU, and all 3 GPUs on this node were fully
utilized by other users' jobs at setup time. Re-check `nvidia-smi` yourself
before running `scripts/serve_qwen.sh`.

## What's ready to go

- **Step 1 (env)**: Miniconda installed at `~/miniconda3` (wasn't present on
  this host). `gdesigner` conda env created (Python 3.10), all CPU-side
  dependencies installed.
- **Step 2 (vLLM)**: installed in a **separate** conda env, `vllm-serve`
  (Python 3.11), not inside `gdesigner`. Reasoning: `vllm==0.6.3.post1` hard-
  pins `torch==2.4.0`, which would fight the `torch==2.3.0+cu121` build that
  `gdesigner`'s `torch_geometric` C-extensions are compiled against. Since
  vLLM is just an HTTP server, there's no reason for it to share a Python env
  with the GDesigner client code. Run it with `scripts/serve_qwen.sh`.
- **Step 3 (.env)**: `.env` created with `BASE_URL=http://localhost:8000/v1`,
  `API_KEY=EMPTY`.
- **Step 4 (dataset)**: MMLU downloaded and verified loadable
  (`datasets/MMLU/data/{dev,val,test}`, 285 dev / 1531 val rows across 57
  topics). Row-count trimming for the smoke test is done at *runtime* via
  the new `--limit_questions` flag on `run_mmlu.py` (see below), not by
  physically slicing the CSVs.
- **Step 5 (smoke test)**: ready to run via `scripts/run_smoke_test.sh` once
  the server from Step 2 is up.
- **Step 7 (gotchas)**: checked against the actual pulled code — see
  "Findings" below, including one real bug that's now fixed.

## Findings from Step 7 (and beyond)

1. **VGAE-vs-code divergence, confirmed.** `grep -rniE "kl_div|logvar|rsample|reparameter" GDesigner/` → no matches. `GDesigner/gnn/gcn.py` is a plain
   2-layer `GCNConv` + a small `MLP`; `graph.py`'s `arun()` combines them via
   a bilinear decode (`logits @ logits.t()`) and min-max normalizes to get
   edge probabilities. No VGAE anywhere.
2. **Topology-freeze claim, confirmed as described.** `Graph.arun()` computes
   `self.spatial_logits` once per query (via the GCN) before the
   `num_rounds` loop, then reuses it for every round within that query -
   frozen mid-episode, recomputed per new query, exactly as instructions.md
   states.
3. **New finding, now fixed: the pulled `GDesigner/llm/gpt_chat.py` did NOT
   speak OpenAI's API.** It POSTed a custom payload
   (`{"name": model, "inputs": {"msg": repr(msg)}}`) with a raw
   `authorization` header straight to `BASE_URL` (not `BASE_URL +
   "/chat/completions"`), and parsed `response_data['data']`. That's built
   for some other internal gateway, not for `vllm serve`'s OpenAI-compatible
   endpoint - as pulled from GitHub, Step 3's `.env` wiring would silently
   never work. Rewrote it to use the `openai` Python SDK's `AsyncOpenAI`
   client (already a pinned dependency) against `BASE_URL`/`API_KEY`, and
   fixed `GPTChat.agen()` to actually forward `max_tokens`/`temperature`
   (previously computed but dropped).
4. **New finding, now fixed: `price.py`'s `cost_count` would crash on the
   `--served-model-name gpt-4o-mini` alias.** `"gpt-4o-mini"` matches the
   `"gpt-4" in model_name` branch but isn't a key in
   `OPENAI_MODEL_INFO["gpt-4"]` → `KeyError` on every single LLM call. Same
   issue for `tiktoken.encoding_for_model` on an unrecognized alias. Both
   now degrade gracefully (cost/token counts become best-effort/zero instead
   of crashing) rather than raising.
5. **New finding, now fixed: `run_mmlu.py` hardcoded `limit_questions =
   153`.** Added a `--limit_questions` CLI flag (default 153, so existing
   full-scale usage is unaffected) so the smoke test can actually use a
   small slice as instructions.md Step 4 asks for.
6. **Environment-only issues found and fixed**, unrelated to GDesigner's own
   code:
   - The repo's `requirements.txt` was UTF-16 encoded and a raw pip-freeze
     from an unrelated dev environment - Windows-only packages
     (`pywin32*`), heavy unused packages never imported anywhere in this
     codebase (`tensorflow`, `keras`, `gradio`, `jupyter`, ...), and a
     version conflict (`protobuf==3.19.5` vs. `google-api-core`'s
     `>=3.20.2`) that made it uninstallable as-is. Replaced with a minimal
     `requirements.txt` derived from this codebase's actual imports (see
     the file's header comment for the full story).
   - `sentence-transformers==2.2.2` (used for node-role embeddings) needs
     `huggingface_hub.cached_download`, removed in newer `huggingface_hub`.
     Pinned `huggingface-hub==0.25.0`, `transformers==4.44.2`,
     `tokenizers==0.19.1` to a mutually-compatible generation.
   - `class_registry` (used by `AgentRegistry`/`LLMRegistry`) needs
     `pkg_resources`, which `setuptools>=81` dropped. Pinned
     `setuptools<81`.
   - `pyg-lib`'s compiled extension needs `GLIBC_2.29`; this host has 2.28
     (`ldd --version`). Skipped it - `torch_sparse` has a clean fallback
     when it's absent (confirmed via its own `try/except ImportError`, the
     only wrinkle being that `OSError` isn't `ImportError`, so pyg-lib had
     to be fully uninstalled rather than left half-broken).
   - vLLM's own PyPI metadata declares `transformers>=4.45.2` with no upper
     bound; an unconstrained install pulls the latest `transformers`, which
     now requires `torch>=2.5` and silently disables itself against vLLM's
     pinned `torch==2.4.0`. Pinned `transformers==4.46.3` in the
     `vllm-serve` env specifically.
   - Latest (unpinned) `vllm` on PyPI pulls `torch==2.13` plus several very
     heavy, irrelevant kernel-compiler extras (tilelang, quack_kernels,
     cutlass_dsl) and ran the shared, nearly-full (`/home` was 97% full)
     disk out of space mid-install. Pinned to `vllm==0.6.3.post1`, the
     release generation that added Qwen2.5 support, which is far lighter.

## Known issue outside the smoke test's scope

`experiments/run_gsm8k.py` and `experiments/run_humaneval.py` currently fail
to import (`ImportError: cannot import name 'VisualLLMRegistry' from
'GDesigner.llm'`) - `GDesigner/tools/reader/readers.py` unconditionally
imports `VisualLLMRegistry`, which is referenced but never defined anywhere
in the pulled repo (upstream dead code, unrelated to anything I changed).
`experiments/run_mmlu.py` - the one instructions.md actually exercises -
does not import `readers.py` and is unaffected. Not fixed here since it's
out of scope for the MMLU smoke test; flagging in case GSM8K/HumanEval come
up later.

## Verified without a GPU

- Full CPU dry-run of `Graph` construction, feature embedding (sentence-
  transformers, now pre-downloaded/cached), the GCN forward pass, and the
  bilinear adjacency decoder - all run cleanly.
- Spun up a throwaway fake OpenAI-compatible HTTP server
  (`http.server`, no extra deps) and ran a **full** `Graph.arun()` cycle
  against it end-to-end - multi-agent spatial/temporal message passing,
  decision node, `postprocess_answer` all worked correctly. This is strong
  evidence the pipeline will work unmodified once pointed at the real vLLM
  server, since vLLM implements the same OpenAI-compatible contract.
- `torch.cuda.is_available()` returns `True` in both envs (driver/CUDA
  runtime is correctly wired up), even though no GPU is currently free for
  actual compute.

## To run once a GPU is free

```bash
# terminal 1
cd /home/rsrimathi/Gdesigner/GDesigner
scripts/serve_qwen.sh          # optionally: scripts/serve_qwen.sh <gpu_index>

# terminal 2 (after terminal 1 reports "Uvicorn running on http://0.0.0.0:8000")
cd /home/rsrimathi/Gdesigner/GDesigner
scripts/run_smoke_test.sh      # optionally: scripts/run_smoke_test.sh <gpu_index>
```

`scripts/serve_qwen.sh` re-prints `nvidia-smi` before launching - check it
and pass a different GPU index as the first argument if GPU 0 isn't the
freest one.
