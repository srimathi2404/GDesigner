```markdown
# G-Designer + Qwen2.5 — Full Setup & Smoke Test

**Goal:** Set up G-Designer from scratch on `fustionstor3`, serve Qwen2.5-7B-Instruct locally on GPU, and run a few small end-to-end test runs (not a full training pass) to confirm the pipeline works: LLM serving → GCN produces adjacency → agents execute over it → accuracy printed.

---

## Step 0 — Check GPU availability

Node `fustionstor3` has 3× RTX 3090 (24GB each), **shared with other users' jobs**. Always check before starting anything, and re-check each session — don't trust old numbers.

```bash
nvidia-smi
```

Pick the GPU with the lowest utilization and most free memory. Examples below assume GPU 0 — swap the index if another GPU is freer.

24GB is enough to serve **Qwen2.5-7B-Instruct in full fp16** (~15GB weights + KV cache) on a single card — no quantization needed. Since GPUs here are shared, compute contention is the more likely slowdown, not VRAM.

---

## Step 1 — Set up the environment

```bash
conda create -n gdesigner python=3.10 -y
conda activate gdesigner

git clone https://github.com/yanweiyue/GDesigner
cd GDesigner
pip install -r requirements.txt
```

Also install vLLM if it isn't already available in this env:

```bash
pip install vllm
```

---

## Step 2 — Install and serve Qwen2.5 on GPU

No manual download step needed — `vllm serve` pulls the model from Hugging Face automatically on first run (cached locally after that).

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --served-model-name gpt-4o-mini \
  --gpu-memory-utilization 0.7
```

Notes:
- `--served-model-name gpt-4o-mini` aliases the served model name so you don't have to edit every script in G-Designer that hardcodes `gpt-4o-mini`.
- `--gpu-memory-utilization 0.7` is conservative on a shared card — vLLM reserves memory upfront based on this fraction, so a too-aggressive value can OOM if another user's job grows. Raise it later once you know the actual free headroom.
- Leave this running in its own terminal/tmux session — it's a long-lived server. Confirm it's up:
  ```bash
  curl http://localhost:8000/v1/models
  ```
- If vLLM's upfront memory reservation keeps causing OOM on this shared card, `llama.cpp`/Ollama serving a GGUF build of Qwen2.5-7B is a more elastic fallback (doesn't pre-reserve, some throughput cost).

---

## Step 3 — Point G-Designer at the local Qwen server

In the `GDesigner` directory, copy the template env file:

```bash
cp template.env .env
```

Edit `.env`:

```
BASE_URL=http://localhost:8000/v1
API_KEY=EMPTY
```

---

## Step 4 — Get a small dataset slice

Download one dataset (MMLU is a good first pick) per the repo's `datasets/` instructions, then cut it down to ~20–30 rows for the smoke test — full splits come later once the pipeline is confirmed working.

---

## Step 5 — Run the smoke test

In a second terminal (keep the vLLM server running in the first):

```bash
conda activate gdesigner
cd GDesigner

CUDA_VISIBLE_DEVICES=0 python experiments/run_mmlu.py \
  --mode FullConnected \
  --batch_size 2 \
  --agent_nums 3 \
  --num_iterations 2 \
  --num_rounds 1 \
  --optimized_spatial
```

This is intentionally tiny — a handful of LLM calls, a couple of optimizer steps — so it should finish in minutes, not hours.

---

## Step 6 — What "it worked" looks like

- No connection-refused errors talking to `localhost:8000` (confirms vLLM + G-Designer are wired together).
- No CUDA OOM (if it happens: lower `--gpu-memory-utilization` on the vLLM server, or lower `--batch_size` on the G-Designer run).
- Logs show a sampled adjacency matrix per query.
- Agents actually execute and produce answers under that topology.
- A final accuracy number prints, even if meaningless on such a small slice — the point is confirming the full loop runs cleanly.

---

## Step 7 — Known gotchas to check in the code

- **Paper-vs-code divergence (already confirmed in earlier analysis):** the G-Designer paper describes a VGAE (mean/logvar heads, reparameterization, KL term), but the actual GNN in `GDesigner/gnn/gcn.py` is a deterministic 2-layer GCN + bilinear decoder. Grep for `kl`, `logvar`, `rsample` to confirm none exist in the commit you pulled.
- **Topology is frozen per query at inference** — check `graph.py`'s spatial-connection construction; LLM outputs shouldn't re-derive the adjacency mid-episode.
- **Training is on-policy REINFORCE, not supervised** — every `--num_iterations` step does a fresh rollout (sample edges → run agents → reward → update). There's no cached dataset reused across steps, which is why LLM call volume scales with iterations, not just dataset size.
- **Model-name / answer-parsing assumptions** — the eval code was tuned against GPT-family output formatting. If accuracy looks suspiciously low with Qwen, check the parser isn't silently failing on Qwen's output format before concluding the topology is bad.
- **Shared-node etiquette** — always set `CUDA_VISIBLE_DEVICES` explicitly; re-check `nvidia-smi` before each run; keep `--gpu-memory-utilization` conservative until you've observed other users' jobs' footprint over time.

---

## Step 8 — After the smoke test passes

Scale up gradually and independently: dataset size → `--num_iterations` → `--agent_nums` → `--num_rounds`. Each is a config change, not new plumbing, once Step 5 runs clean end-to-end.
```