import os
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset

# Directory holding the bundled math data files (aime24.jsonl, aime25.jsonl),
# resolved relative to this file so the package is self-contained.
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please check the path / mount / permissions."
        )
    ds = load_dataset("json", data_files={"train": path}, split="train")
    return ds


def load_data(name: str) -> List[Dict[str, Any]]:

    if name == "AIME24":
        path = os.path.join(_DATA_DIR, "aime24.jsonl")
        dataset = _load_local_jsonl(path)
        return [{"problem": x["problem"], "groundtruth": x["answer"]} for x in dataset]

    elif name == "AIME25":
        path = os.path.join(_DATA_DIR, "aime25.jsonl")
        dataset = _load_local_jsonl(path)
        return [{"problem": x["problem"], "groundtruth": x["answer"]} for x in dataset]

    elif name == "DAPO-Math-17k":
        # Use a local cache if present, otherwise pull from the Hub.
        cache_path = os.path.join(_DATA_DIR, "cleaned_data.json")
        if os.path.exists(cache_path):
            return json.load(open(cache_path))
        else:
            dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
            data = dataset.to_list()
            transformed = {}
            for record in data:
                problem = record["prompt"][0]["content"].replace(
                    "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n",
                    "",
                ).replace('\n\nRemember to put your answer on its own line after "Answer:".', "")
                groundtruth = record["reward_model"]["ground_truth"]
                transformed[problem] = groundtruth
            random.seed(42)
            transformed = [{"problem": k, "groundtruth": v} for k, v in transformed.items()]
            random.shuffle(transformed)
            os.makedirs("data/math/dataset", exist_ok=True)
            json.dump(transformed, open("data/math/dataset/DAPO-Math-17k.json", "w"), indent=2)
            return transformed

    raise ValueError(f"Unsupported dataset: {name}")
