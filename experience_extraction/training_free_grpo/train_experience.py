"""
train_experience.py — Experience summarization loop (training-free GRPO style).

This is the experience-extraction stage of the self-evolution pipeline in
"Rethinking Continual Experience Internalization for Self-Evolving LLM Agents".
For each batch of queries:

  1. Rollout      : the agent (utu.SimpleAgent + tools) solves a batch of queries,
                    GRPO-style (grpo_n rollouts per query). Always on the local model.
  2. Verify       : an LLM judge scores each rollout against the ground truth.
  3. Experience   : ExperienceUpdater distills the scored trajectories into a
                    principle-level experience pool (rollout summary -> per-query
                    critique -> group update -> batch update / consolidation).

Two domains are supported via --domain: ``web`` (web-reasoning QA with search
tools) and ``math`` (math problem solving).

The rollout always runs on a local model (served by vLLM). The experience-update
/ verify LLM is selected by ``--experience_backend``:

  * ``deepseek`` (the paper's main setting) — the experience update runs on an
    external API (DeepSeek) read from UTU_LLM_* in .env, while rollout stays local.
  * ``local`` — the experience update reuses the same local vLLM as the rollout
    (fully local, self-generated experience; no external API needed).

Usage (start a vLLM server in another terminal first; a large --max-model-len is
recommended to fit the long experience-update prompts):

    bash training_free_grpo/start_vllm.sh 0,1,2,3 8100

then (paper's main setting — rollout local, experience update on DeepSeek; set
UTU_LLM_* / UTU_LLM_API_KEY in .env to your DeepSeek endpoint first):

    python -m training_free_grpo.train_experience \
        --mode agent --domain web --experience_backend deepseek \
        --experiment_name qwen3_web \
        --data_file data/train/data_15k_shuffle.jsonl \
        --epochs 3 --batchsize 4 --grpo_n 5 --max_pool_size 50 \
        --local_model_name qwen3-4b --local_base_url http://localhost:8100/v1

Outputs land under ``data/<domain>/train/<experiment_name>/`` — one ``step_k/`` per
batch, each holding the rollouts and intermediate experience artifacts, and
``step_{k+1}/experiences.json`` is the experience pool fed into the next step.
"""

import argparse
import asyncio
import copy
import json
import os
import random

os.environ["AGENTS_MAX_TURNS"] = "50"
os.environ["MAX_TURNS"] = "50"

# ---------------------------------------------------------------------------
# Step 0: set the env vars that utu requires at import time.
#
# utu/__init__.py runs EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL"]) on
# import, and utu/utils/env.py loads .env with load_dotenv(override=True). These
# setdefaults only guard against a missing/blank .env so the import-time assertion
# passes; the effective values are set per backend in main().
# ---------------------------------------------------------------------------
os.environ.setdefault("UTU_LLM_TYPE", "openai")
os.environ.setdefault("UTU_LLM_MODEL", "local")
os.environ.setdefault("UTU_LLM_BASE_URL", "http://localhost:8100/v1")
os.environ.setdefault("UTU_LLM_API_KEY", "not-needed")

# ---------------------------------------------------------------------------
# Step 1: LLM routing helpers.
#
# The experience-update / verify LLM is `training_free_grpo.llm.LLM`, used by
# ExperienceUpdater and verify_func. Depending on --experience_backend, main()
# either patches LLM onto the local vLLM (local backend) or leaves it reading
# UTU_LLM_* from the environment (deepseek backend). Patching __init__ on the
# class object affects every reference to LLM regardless of import order.
# ---------------------------------------------------------------------------
import openai as _openai
import training_free_grpo.main as _main_module
import training_free_grpo.llm as _llm_module

# Local-vLLM config, filled from CLI args in main().
_local_config = {
    "model": "qwen3-4b",
    "base_url": "http://localhost:8100/v1",
    "api_key": "not-needed",
}


def _patched_llm_init(self):
    """LLM.__init__ replacement: skip env assertions, connect to the local vLLM."""
    self.model_name = _local_config["model"]
    self.client = _openai.OpenAI(
        api_key=_local_config["api_key"],
        base_url=_local_config["base_url"],
    )


class _LocalLLM(_llm_module.LLM):
    """An LLM pinned to the local vLLM, used to keep prompt-mode rollout on the
    local model while the experience update still uses the external API."""

    def __init__(self):
        self.model_name = _local_config["model"]
        self.client = _openai.OpenAI(
            api_key=_local_config["api_key"],
            base_url=_local_config["base_url"],
        )


from training_free_grpo.main import rollout_dataset, load_rollouts  # noqa: E402
from utu.agents import SimpleAgent  # noqa: E402
from utu.config import ConfigLoader  # noqa: E402

random.seed(42)


def _load_jsonl_file(filepath: str) -> list[dict]:
    """Load a JSONL file, mapping field names to problem/groundtruth."""
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            item = {
                "id": row.get("id", len(data)),
                "problem": row.get("problem") or row.get("question", ""),
                "groundtruth": row.get("groundtruth") or row.get("answer", ""),
            }
            # Keep any other fields from the original record.
            for k, v in row.items():
                if k not in ("id", "problem", "groundtruth", "question", "answer"):
                    item[k] = v
            data.append(item)
    return data


def _build_local_agent(config_name: str, args) -> SimpleAgent:
    """Build a SimpleAgent pointed at the local vLLM server."""
    config = ConfigLoader.load_agent_config(config_name)
    config.model.model_settings.temperature = args.rollout_temperature
    # Override the three model_provider fields (model_dump's exclude_none=True
    # ensures these non-None values take precedence over env defaults).
    config.model.model_provider.model = args.local_model_name
    config.model.model_provider.base_url = args.local_base_url
    config.model.model_provider.api_key = args.local_api_key
    return SimpleAgent(config=config)


async def main(args):
    # Fill the local-vLLM config (used by the rollout, and by the experience
    # update when --experience_backend local).
    _local_config["model"] = args.local_model_name
    _local_config["base_url"] = args.local_base_url
    _local_config["api_key"] = args.local_api_key

    # ---- experience / verify LLM backend ----
    if args.experience_backend == "local":
        # Everything (rollout, verify, experience update, in-agent tool LLM) runs
        # on the local vLLM. Patch LLM and point UTU_LLM_* at the local server.
        _llm_module.LLM.__init__ = _patched_llm_init
        os.environ["UTU_LLM_MODEL"] = args.local_model_name
        os.environ["UTU_LLM_BASE_URL"] = args.local_base_url
        os.environ["UTU_LLM_API_KEY"] = args.local_api_key
        print("[Experience]  update/verify LLM -> local vLLM")
    else:  # deepseek — the paper's main setting
        # Rollout stays on the local vLLM (pinned below); verify and the
        # experience update use the external API read from UTU_LLM_* (.env, e.g.
        # DeepSeek). Optionally override those here without editing .env.
        if args.experience_model:
            os.environ["UTU_LLM_MODEL"] = args.experience_model
        if args.experience_base_url:
            os.environ["UTU_LLM_BASE_URL"] = args.experience_base_url
        if args.experience_api_key:
            os.environ["UTU_LLM_API_KEY"] = args.experience_api_key
        # Pin prompt-mode rollout to the local model (agent mode is pinned via the
        # agent config); leave LLM unpatched so experience/verify use the API.
        _main_module.LLM = _LocalLLM
        print(f"[Experience]  update/verify LLM -> external API ({os.environ.get('UTU_LLM_MODEL')})")

    # Optionally override the search API keys from the environment without
    # touching .env (useful when a stale .env key has expired).
    _serper_override = os.environ.get("SERPER_API_KEY_OVERRIDE")
    if _serper_override:
        os.environ["SERPER_API_KEY"] = _serper_override
        print("[env] SERPER_API_KEY overridden from SERPER_API_KEY_OVERRIDE")
    _jina_override = os.environ.get("JINA_API_KEY_OVERRIDE")
    if _jina_override:
        os.environ["JINA_API_KEY"] = _jina_override
        print("[env] JINA_API_KEY overridden from JINA_API_KEY_OVERRIDE")

    # Domain config (lazy import; LLM backend is already configured above).
    if args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.web.experience import ExperienceUpdater

        config_name = "simple/base_search.yaml"
    elif args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.math.experience import ExperienceUpdater

        config_name = "simple/math_agent.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    # Create the experiment directory.
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Build the rollout agent / LLM (always local).
    if args.mode == "prompt":
        worker_agent = None
        print(f"[Rollout]     prompt mode -> local {args.local_model_name} @ {args.local_base_url}")
    elif args.mode == "agent":
        worker_agent = _build_local_agent(config_name, args)
        await worker_agent.build()
        print(f"[Rollout]     agent mode  -> local {args.local_model_name} @ {args.local_base_url}")
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset.
    if args.data_file:
        train_data = _load_jsonl_file(args.data_file)
        print(f"Loaded {len(train_data)} records from {args.data_file}")
    elif args.dataset:
        train_data = load_data(args.dataset)
        print(f"Loaded {len(train_data)} records from dataset {args.dataset}")
    else:
        raise ValueError("Either --data_file or --dataset must be provided")
    # dataset_truncate is the number of records randomly sampled per epoch (a
    # fresh sample each epoch), not just the first N records.
    if args.dataset_truncate is None:
        args.dataset_truncate = len(train_data)
    print(f"- {len(train_data)} total; sampling {args.dataset_truncate} per epoch (fresh per epoch)")
    assert args.dataset_truncate % args.batchsize == 0, (
        f"dataset_truncate ({args.dataset_truncate}) must be divisible by batchsize ({args.batchsize})"
    )

    # Load stats (supports resuming).
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        with open(stats_filename) as f:
            stats = json.load(f)
    else:
        stats = {}

    # Main loop.
    for epoch in range(args.epochs):
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Sample dataset_truncate fresh records from train_data each epoch. The
        # sample is cached to a file so the epoch can resume; epochs sample
        # independently of each other.
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"[freshsample] Loaded {len(shuffled_data)} pre-sampled records from {shuffled_filename}")
        else:
            print(f"[freshsample] epoch {epoch}: random.sample {args.dataset_truncate} from {len(train_data)} ...")
            shuffled_data = random.sample(train_data, args.dataset_truncate)
            shuffled_data = copy.deepcopy(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "complete": False,
                }
            elif stats[f"step_{step}"]["complete"]:
                continue

            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            # Current batch.
            batch_data = copy.deepcopy(
                shuffled_data[
                    batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize
                ]
            )

            # Load existing rollouts (resume support).
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)

            # Load the experience pool (step 0 has none).
            if step > 0:
                experience_filename = os.path.join(
                    experiment_dir, f"step_{step}/experiences.json"
                )
                with open(experience_filename) as f:
                    experiences = json.load(f)
            else:
                experiences = {}

            # Format the experience pool and inject it into the prompt.
            formatted_experiences = "\n".join(
                [f"[{i}]. {e}" for i, e in experiences.items()]
            )
            formatted_batch_data = [
                {
                    "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                        experiences=(
                            formatted_experiences if formatted_experiences else "None"
                        ),
                        problem=each["problem"],
                    )
                    if experiences
                    else each["problem"],
                    **each,
                }
                for each in batch_data
            ]

            # GRPO: repeat each query grpo_n times.
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n

            # Rollout.
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )
            stats[f"step_{step}"]["rollout"] = rollout_stats

            # Experience update (uses the backend configured above).
            next_step_dir = os.path.join(experiment_dir, f"step_{step + 1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(
                next_step_dir, "experiences.json"
            )
            if os.path.exists(next_experience_filename):
                print(
                    f"Experiences already exist for step {step}, skipping"
                )
            else:
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts,
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=(args.given_ground_truth == "True"),
                    max_pool_size=args.max_pool_size,
                )
                with open(next_experience_filename, "w") as f:
                    json.dump(new_experiences, f, indent=2)
                print(
                    f"Saved {len(new_experiences)} experiences to {next_experience_filename}"
                )

            # Save stats.
            stats[f"step_{step}"]["complete"] = True
            with open(stats_filename, "w") as f:
                json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training-free GRPO experience summarization (local rollout)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompt",
        choices=["prompt", "agent"],
        help="prompt: plain LLM inference; agent: SimpleAgent + tool calls",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="web",
        choices=["web", "math"],
    )
    parser.add_argument(
        "--experience_backend",
        type=str,
        default="local",
        choices=["local", "deepseek"],
        help="Where the experience-update / verify LLM runs. 'deepseek' (the "
             "paper's main setting) uses the external API from UTU_LLM_* in .env; "
             "'local' reuses the local vLLM.",
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="JSONL file path (fields: question/answer or problem/groundtruth)",
    )
    parser.add_argument("--dataset_truncate", type=int, default=None)
    parser.add_argument("--given_ground_truth", type=str, default="True")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--rollout_concurrency", type=int, default=5)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_max_tokens", type=int, default=8192)
    parser.add_argument("--task_timeout", type=float, default=3600)
    parser.add_argument("--max_pool_size", type=int, default=50)
    parser.add_argument(
        "--local_model_name",
        type=str,
        default="qwen3-4b",
        help="vLLM served model name (used for rollout)",
    )
    parser.add_argument(
        "--local_base_url",
        type=str,
        default="http://localhost:8100/v1",
        help="local vLLM server URL (used for rollout)",
    )
    parser.add_argument(
        "--local_api_key",
        type=str,
        default="not-needed",
        help="local vLLM API key (usually not needed)",
    )
    parser.add_argument(
        "--experience_model",
        type=str,
        default=None,
        help="(deepseek backend) override UTU_LLM_MODEL for the experience LLM",
    )
    parser.add_argument(
        "--experience_base_url",
        type=str,
        default=None,
        help="(deepseek backend) override UTU_LLM_BASE_URL for the experience LLM",
    )
    parser.add_argument(
        "--experience_api_key",
        type=str,
        default=None,
        help="(deepseek backend) override UTU_LLM_API_KEY for the experience LLM",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
