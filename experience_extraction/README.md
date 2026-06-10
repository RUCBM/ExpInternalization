<div align="center">

# Rethinking Continual Experience Internalization for Self-Evolving LLM Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.04703)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/RUCBM/ExpInternalization)

</div>

**Experience Extraction** — the stage that turns an agent's interaction trajectories into a compact, reusable **experience pool**, which is later internalized into the model parameters to drive self-evolution.

Trajectories are rolled out by a **local** model (served by vLLM); the experience summary/update is produced either by a stronger **external model** (DeepSeek — the paper's main setting) or by the same local model (self-generated). This is the agent-rollout + experience-summarization part of our pipeline; it is built on top of [youtu-agent](https://github.com/TencentCloudADP/youtu-agent) and supports both **web reasoning** and **math** tasks.

## Highlights

- 🧩 **Principle-level experience.** Trajectories are distilled into concise, transferable strategies (rather than instance-specific traces), which are far more durable across self-evolution iterations.
- 🔁 **Self-maintaining pool.** A bounded experience pool with ADD / UPDATE / DELETE operations and automatic LLM-based consolidation keeps experience compact and non-redundant across rounds.
- 🤖 **Flexible experience backend.** Rollout always runs on the local model; the experience update can use a stronger external API (DeepSeek — the paper's main setting) or the same local model (fully self-generated, no external API). Switch with `--experience_backend`.
- 🌐 **Two domains out of the box.** `--domain web` (search-tool agents) and `--domain math`.

## How it works

For each batch of queries, `train_experience.py` runs a GRPO-style loop:

1. **Rollout** — the agent (`utu.SimpleAgent` + tools) solves each query `grpo_n` times.
2. **Verify** — an LLM judge scores each rollout against the ground truth.
3. **Experience update** — `ExperienceUpdater` distills the scored trajectories into the experience pool in four steps: per-rollout summary → per-query critique → group update (ADD/UPDATE/DELETE) → batch reconcile & consolidate to a fixed pool size.

The pool produced at each step is fed as context into the next step, so the agent and its experience co-evolve.

## Installation

Requires Python ≥ 3.12.

```bash
# install dependencies (uv recommended)
uv sync                       # or:  pip install -r requirements.txt

# configure keys for the web search tools (not needed for --domain math)
cp .env.example .env
# then set in .env:
#   SERPER_API_KEY / JINA_API_KEY      web search tools  (https://serper.dev, https://jina.ai/reader)
#   UTU_LLM_* (incl. UTU_LLM_API_KEY)  the experience-update API for the
#                                      `deepseek` backend (DeepSeek by default)
```

## Quick Start

**1. Serve the student model with vLLM** (a 4B model fits on 1–2 GPUs):

```bash
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
  bash training_free_grpo/start_vllm.sh 0,1 8100
```

**2. Run experience extraction** — the paper's main setting is local rollout with
DeepSeek-generated experience (`--experience_backend deepseek`). Web reasoning:

```bash
python -m training_free_grpo.train_experience \
    --mode agent --domain web --experience_backend deepseek \
    --experiment_name qwen3_web \
    --data_file data/train/data_15k_shuffle.jsonl \
    --dataset_truncate 100 --epochs 3 --batchsize 4 --grpo_n 5 \
    --rollout_concurrency 128 --task_timeout 1800 --max_pool_size 50 \
    --local_model_name qwen3-4b --local_base_url http://localhost:8100/v1
```

With `--dataset_truncate 100 --batchsize 4` (25 batches/epoch) over 3 epochs this
produces **75 steps**, i.e. `step_0 … step_75/experiences.json`.

Math (no search keys needed):

```bash
python -m training_free_grpo.train_experience \
    --mode agent --domain math --experience_backend deepseek \
    --experiment_name qwen3_math \
    --dataset DAPO-Math-17k \
    --dataset_truncate 100 --epochs 3 --batchsize 4 --grpo_n 5 \
    --rollout_concurrency 64 --task_timeout 1800 --max_pool_size 50 \
    --local_model_name qwen3-4b --local_base_url http://localhost:8100/v1
```

To run everything on the local model instead (no external API), pass
`--experience_backend local`. The one-shot script below serves vLLM and runs the
web loop in a single command:

```bash
bash scripts/run_experience_summarize.sh
```

For multi-iteration self-evolution, re-launch with the next-iteration model serving vLLM; each round regenerates trajectories and refreshes the pool.

## Outputs

Results are written under `data/<domain>/train/<experiment_name>/`. Each `step_k/` holds the rollouts and the intermediate experience artifacts, and `step_{k+1}/experiences.json` is the experience pool produced at that step. A runnable example is provided in `data/examples/`.

## Acknowledgements & License

The agent runtime, tools, and configs (`utu/`, `configs/`) are vendored from [youtu-agent](https://github.com/TencentCloudADP/youtu-agent) (`training_free_GRPO` branch), © Tencent, MIT-licensed — see `LICENSE.youtu-agent`.

Our own code (`training_free_grpo/`, `scripts/`) is released under the MIT License — see `LICENSE`. `NOTICE` summarizes the third-party attribution.
