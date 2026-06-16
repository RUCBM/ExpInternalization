import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
import react_agent
from react_agent import MultiTurnReactAgent
from react_agent_step_wise import StepWiseReactAgent
import time
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=1)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    # Agent / injection mode for inference:
    #   plain       - no inference-time experience (internalized model only)
    #   global    - inject the WHOLE experience pool into the system prompt,
    #               fixed for the entire trajectory (global injection)
    #   step_wise - inject step-wise-selected experience at each turn via a
    #               selector model (step-wise injection)
    parser.add_argument("--agent", type=str, default="plain",
                        choices=["plain", "global", "step_wise"])
    parser.add_argument("--exp_path", type=str, default="experiences.json",
                        help="(global / step_wise) experience pool JSON to inject")
    parser.add_argument("--retrieval_model", type=str, default="",
                        help="(step_wise) model name used to select experience per step")
    parser.add_argument("--retrieval_ports", type=int, nargs="+", default=[8001],
                        help="(step_wise) vLLM ports serving the retrieval/selector model")
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))

    model_dir = os.path.join(output_base, f"{model_name}_sglang")
    # Use the dataset file's basename (without .json/.jsonl) as the directory name.
    dataset_name = os.path.splitext(os.path.basename(args.dataset.rstrip("/")))[0]
    dataset_dir = os.path.join(model_dir, dataset_name)

    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_queries_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    
    # One planning port per vLLM server (here: 4 servers).
    planning_ports = [7001, 7002, 7003, 7004]
    
    # Round-robin state
    planning_rr_idx = 0
    summary_rr_idx = 0
    # Sticky assignment per question
    question_to_ports = {}
    
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                # Ensure sticky and balanced port assignment per unique question
                if question not in question_to_ports:
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1
                planning_port = question_to_ports[question]
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": planning_port,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            },
            'model_type': 'qwen_dashscope'
        }

        tools = ["search", "visit", "google_scholar", "PythonInterpreter"]
        if args.agent == "step_wise":
            test_agent = StepWiseReactAgent(
                llm=llm_cfg,
                function_list=tools,
                retrieval_model=args.retrieval_model,
                retrieval_ports=args.retrieval_ports,
                exp_path=args.exp_path,
            )
        else:
            if args.agent == "global":
                # Global injection: prepend the WHOLE experience pool to the
                # system prompt, fixed for the entire trajectory (c_glob = [x; E]).
                with open(args.exp_path, "r", encoding="utf-8") as f:
                    pool = json.load(f)
                accumulated = "\n".join(f"{k}: {v}" for k, v in pool.items())
                react_agent.SYSTEM_PROMPT = (
                    react_agent.SYSTEM_PROMPT
                    + "\n\nSolve the input problem by leveraging relevant insights "
                    + "from your accumulated experiences.\n\n"
                    + "**ACCUMULATED EXPERIENCES:**\n" + accumulated
                )
            test_agent = MultiTurnReactAgent(
                llm=llm_cfg,
                function_list=tools,
            )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model
                ): task for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = future.result()
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]"
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")