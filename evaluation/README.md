# Web Evaluation

Evaluation code for *Rethinking Continual Experience Internalization for Self-Evolving LLM Agents* ([paper](https://arxiv.org/abs/2606.04703)).

A ReAct agent with web tools (search / visit / scholar / python) is run on the web benchmarks **WebWalkerQA**, **GAIA-Text-103**, and **BrowseComp-ZH**; an LLM judge then scores the predictions. Built on the [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) inference and evaluation code.

## Setup

```bash
pip install -r requirements.txt        # vLLM, qwen-agent, transformers, litellm, ...
cp .env.example .env                   # then fill in the values below
```

Required in `.env`:

| variable | purpose |
|---|---|
| `MODEL_PATH` | model under evaluation (HF id or local dir) |
| `SERPER_KEY_ID` | web search ([serper.dev](https://serper.dev)) |
| `JINA_API_KEYS` | web page reader ([jina.ai](https://jina.ai/reader)) |
| `API_KEY`, `API_BASE`, `SUMMARY_MODEL_NAME` | endpoint used to summarize visited pages |
| `OPENAI_API_KEY`, `OPENAI_API_BASE` | endpoint for the judge model (scoring) |

## Step 1 — inference

Serve the model with vLLM and run the agent. `run_react_infer.sh` does both from `.env`:

```bash
bash run_react_infer.sh
```

Or run the agent directly against an already-served vLLM endpoint. Three inference
modes are selected with `--agent`:

```bash
# plain: no inference-time experience (internalized model)
python -u run_multi_react.py --agent plain \
    --dataset eval_data/webwalkerqa.jsonl --output outputs/webwalker \
    --model $MODEL_PATH --roll_out_count 3 --max_workers 30

# global: prepend the whole experience pool to the system prompt (global injection)
python -u run_multi_react.py --agent global --exp_path experiences.json \
    --dataset eval_data/webwalkerqa.jsonl --output outputs/webwalker_global \
    --model $MODEL_PATH --roll_out_count 3 --max_workers 30

# step_wise: select experience per step with a selector model (step-wise injection)
python -u run_multi_react.py --agent step_wise --exp_path experiences.json \
    --retrieval_model $SELECTOR_MODEL --retrieval_ports 8001 \
    --dataset eval_data/webwalkerqa.jsonl --output outputs/webwalker_stepwise \
    --model $MODEL_PATH --roll_out_count 3 --max_workers 30
```

Output: `<output>/<model>_sglang/<dataset>/iter{1,2,3}.jsonl`, one line per query with `question`, `answer`, `prediction`.

## Step 2 — score with the judge

```bash
cd judge
python evaluate_deepsearch_official.py \
    --input_folder <path to the iter{1,2,3}.jsonl dir> \
    --dataset webwalker          # webwalker | gaia | browsecomp_zh
```

`--dataset` sets the judge prompt and judge model (webwalker/gaia → Qwen2.5-72B-Instruct; browsecomp_zh → GPT-4o). It reads `iter{1,2,3}.jsonl` and reports per-round and averaged accuracy. Judge prompts are in `judge/prompt.py`.

## Data

`eval_data/` contains the three benchmarks (each line: `question`, `answer`):

| file | benchmark | # |
|---|---|---|
| `webwalkerqa.jsonl` | WebWalkerQA | 680 |
| `gaia_text_103.jsonl` | GAIA-Text-103 | 103 |
| `browsecomp_zh.jsonl` | BrowseComp-ZH | 289 |

Redistributed from their original public releases; follow each benchmark's license.

## License

Code under `react_agent_step_wise.py`, `prompt_step_wise.py`, and our tool/eval changes is MIT (`LICENSE`). The inference framework, tools, and `judge/` are from [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) (Apache 2.0, `LICENSE.deepresearch`); URL filtering is adapted from MiroThinker; the agent uses [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). See `NOTICE`.
