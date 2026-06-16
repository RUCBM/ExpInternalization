# Experience Extraction

Experience-extraction code for *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* ([paper](https://arxiv.org/abs/2606.04703)).

This stage turns an agent's interaction trajectories into a compact, reusable **experience pool** (later internalized into the model). Trajectories are rolled out by a **local** model (served by vLLM); the experience summary/update is produced either by a stronger external model (**DeepSeek** — the paper's main setting) or by the same local model. Built on [youtu-agent](https://github.com/TencentCloudADP/youtu-agent) (`training_free_GRPO` branch); supports `--domain web` and `--domain math`.

## How it works

`train_experience.py` runs a GRPO-style loop. Per batch:

1. **Rollout** — the agent (`utu.SimpleAgent` + tools) solves each query `grpo_n` times.
2. **Verify** — an LLM judge scores each rollout against the ground truth.
3. **Experience update** — `ExperienceUpdater` distills the scored trajectories into the pool: per-rollout summary → per-query critique → group update (ADD/UPDATE/DELETE) → batch reconcile & consolidate to a fixed pool size.

The pool from each step is fed into the next step. `--experience_backend deepseek` (default-recommended) runs the update on DeepSeek; `--experience_backend local` runs it on the local model (fully self-generated).

## Setup

Requires Python ≥ 3.12.

```bash
uv sync                         # or: pip install -r requirements.txt
cp .env.example .env            # then fill in the values below
```

Required in `.env`:

| variable | purpose |
|---|---|
| `SERPER_API_KEY`, `JINA_API_KEY` | web search / page reader (web domain only) |
| `UTU_LLM_*` (incl. `UTU_LLM_API_KEY`) | experience-update API for `--experience_backend deepseek` (DeepSeek by default) |

## Usage

Serve the local rollout model with vLLM (a 4B model fits on 1–2 GPUs):

```bash
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 bash training_free_grpo/start_vllm.sh 0,1 8100
```

Run experience extraction — **web** domain (paper's main setting, DeepSeek experience):

```bash
python -m training_free_grpo.train_experience \
    --mode agent --domain web --experience_backend deepseek \
    --experiment_name qwen3_web \
    --data_file data/train/data_15k_shuffle.jsonl \
    --dataset_truncate 100 --epochs 3 --batchsize 4 --grpo_n 5 \
    --rollout_concurrency 128 --task_timeout 1800 --max_pool_size 50 \
    --local_model_name qwen3-4b --local_base_url http://localhost:8100/v1
```

`--dataset_truncate 100 --batchsize 4` (25 batches/epoch) × 3 epochs → **75 steps** (`step_0 … step_75/experiences.json`).

**math** domain (no search keys needed): same command with `--domain math --dataset DAPO-Math-17k`.

To run the experience update on the local model instead, pass `--experience_backend local`. The all-local web loop can also be launched in one command with `bash scripts/run_experience_summarize.sh`.

## Outputs

Under `data/<domain>/train/<experiment_name>/`: each `step_k/` holds the rollouts and intermediate artifacts; `step_{k+1}/experiences.json` is the experience pool produced at that step. A runnable example is in `data/examples/`.

## License

Our code (`training_free_grpo/`, `scripts/`) is MIT (`LICENSE`). The vendored runtime (`utu/`, `configs/`) is from [youtu-agent](https://github.com/TencentCloudADP/youtu-agent), © Tencent, MIT (`LICENSE.youtu-agent`). See `NOTICE`.
