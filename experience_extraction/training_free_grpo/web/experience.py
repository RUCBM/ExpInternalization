import json
import copy
import os
import re

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from training_free_grpo.llm import LLM
from training_free_grpo.web.prompts import (
    SINGLE_QUERY_CRITIQUE_TEMPLATE_SP,
    SINGLE_QUERY_CRITIQUE_TEMPLATE_UP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP,
    GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP,
    EXPERIENCE_CONSOLIDATION_TEMPLATE_SP,
    EXPERIENCE_CONSOLIDATION_TEMPLATE_UP,
)


class ExperienceUpdater:
    def __init__(self):
        self.llm = LLM()
    
    def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True, max_pool_size=50):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth
        )

        # 2. Generate critique for each query
        new_experiences = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts, 
            experiences=experiences,
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth
        )

        # 3. group update experiences
        critiques = self._group_update(
            experiences=experiences, 
            new_experiences=new_experiences, 
            save_dir=save_dir,
            max_workers=max_workers
        )

        # 4. batch update experiences
        new_experiences = self._batch_update(
            experiences=experiences,
            critiques=critiques,
            save_dir=save_dir,
            max_pool_size=max_pool_size
        )

        # 5. assign new experience IDs
        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences


    def _single_rollout_summary(
        self,
        rollouts, 
        save_dir, 
        max_workers,
        given_ground_truth=True
    ):
        # check file existence
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single rollout summary")
                    print("- File exists, loaded from:", filename)
                    return results

        # group by problems
        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and len(each["trajectories"]) > 0:
                problems_to_rollouts[each["problem"]].append(each)
        results = defaultdict(list)

        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts_to_process.extend(rollouts)
            else:
                all_rollouts_to_process.extend(rollouts)

        def process(cur):
            try:
                up = SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP.format(
                    task=cur["problem"],
                    trajectory=cur["trajectories"][0]["trajectory"], 
                    answer=cur["groundtruth"] if given_ground_truth else "[REDACTED]"
                )
                response = self.llm.chat(
                    [
                        {"role": "system", "content": SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP},
                        {"role": "user", "content": up}
                    ]
                )
                return {"trajectory_summary": response, **cur}
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        # parallel running
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rollout = {executor.submit(process, cur): cur for cur in all_rollouts_to_process}
            for future in tqdm(
                as_completed(future_to_rollout), total=len(all_rollouts_to_process), desc="Single rollout summary"
            ):
                result = future.result()
                if result is not None:
                    problem = result["problem"]
                    results[problem].append(result)

        # write to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _single_query_critique(
        self,
        problem_to_summarized_rollouts, 
        experiences, 
        save_dir, 
        max_workers, 
        max_operations=1,
        given_ground_truth=True
    ):
        # check file existence
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single query critique")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts.append(rollouts)
            else:
                all_rollouts.append(rollouts)

        def process(rollouts_per_problem):
            try:
                problem = rollouts_per_problem[0]["problem"]
                answer = rollouts_per_problem[0]["groundtruth"]
                formatted_trajectories = "\n\n".join([
                    f"Attempt {i+1} (Answer {'correct' if each['reward'] else 'wrong'}):\n{each['trajectory_summary']}"
                    for i, each in enumerate(rollouts_per_problem)
                ])
                up = SINGLE_QUERY_CRITIQUE_TEMPLATE_UP.format(
                    question=problem,
                    answer=answer if given_ground_truth else "[REDACTED]",
                    attempts=formatted_trajectories,
                )
                response = self.llm.chat(
                    [
                        {"role": "system", "content": SINGLE_QUERY_CRITIQUE_TEMPLATE_SP},
                        {"role": "user", "content": up}
                    ]
                )
                # response = response.split("```json")[-1].split("```")[0]
                # extract experiences from the response
                pattern = re.compile(r"<Experiences>\s*(.*?)\s*</Experiences>",re.DOTALL | re.IGNORECASE)
                match = pattern.search(response)
                experiences = match.group(1).strip() if match else ""
                return {"rollouts": rollouts_per_problem, "critique": response, "experiences": experiences}
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        # parallel running
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(process, rollouts_per_problem): rollouts_per_problem
                for rollouts_per_problem in all_rollouts
            }
            for future in tqdm(as_completed(future_to_case), total=len(all_rollouts), desc="Single query critique"):
                result = future.result()
                if result is not None:
                    results.append(result)

        # write results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _group_update(
        self,
        experiences, 
        new_experiences, 
        save_dir,
        max_workers=16
    ):
        # check file existence
        filename = os.path.join(save_dir, "group_update.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Group update")
                    print("- File exists, loaded from:", filename)
                    return results
        
        def process(new_experience):
            try:
                formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ]) if experiences else "None"
                up = GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    existing_experiences=formatted_experiences,
                    new_experiences=new_experience["experiences"],
                )
                response = self.llm.chat(
                    [
                        {"role": "system", "content": GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP},
                        {"role": "user", "content": up}
                    ]
                )
                # parse response
                response = response.split("```json")[-1].split("```")[0]
                operations = json.loads(response)
                return {"operations": operations, **new_experience}
            except Exception as e:
                print(f"Warning: failed in group update, {e}")
                return None
        
        # parallel running
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(process, new_experience): new_experience
                for new_experience in new_experiences
            }
            for future in tqdm(as_completed(future_to_case), total=len(new_experiences), desc="Group update"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # write results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _batch_update(
        self,
        experiences,
        critiques,
        save_dir,
        max_retries=3,
        max_pool_size=50
    ):
        print("Batch update")
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename):
            results = json.load(open(filename))
            print("- File exists, loaded from:", filename)
            return results.get("new_experiences", results)
        
        # collect operations
        all_operations = []
        for each in critiques:
            all_operations.extend(each["operations"])
        print("- Num of operations to process:", len(all_operations))

        # use LLM to get the revision plan
        revision_plan = []
        for _ in range(max_retries):
            try:
                up = BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP.format(
                    experiences_and_operations=self._format_exp_and_ops(experiences, all_operations),
                    current_pool_size=len(experiences),
                    max_pool_size=max_pool_size
                )
                response = self.llm.chat(
                    [
                        {"role": "system", "content": BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP.replace(
                            "{max_pool_size}", str(max_pool_size))},
                        {"role": "user", "content": up}
                    ]
                )
                revision_plan = json.loads(response.split("```json")[-1].split("```")[0])
                break
            except Exception:
                print("Warning: failed to decode in updating general experiences")

        # apply revision plan to get new experiences
        max_ID = len(experiences)
        new_experiences = copy.deepcopy(experiences)
        for plan in revision_plan:
            operation = plan.get("operation", "ADD")
            content = plan.get("content", "")
            target_id = plan.get("id", None)
            if not content:
                continue

            if operation == "ADD":
                new_experiences[f"{max_ID}"] = content
                max_ID += 1
            elif operation == "UPDATE":
                if target_id in new_experiences:
                    new_experiences[target_id] = content
                else:
                    # directly add new experience
                    new_experiences[f"{max_ID}"] = content
                    max_ID += 1
            elif operation == "DELETE":
                if target_id in new_experiences:
                    del new_experiences[target_id]
        # filter out non-string values (Qwen3-4B may output [] or {} instead of strings)
        invalid_keys = [k for k, v in new_experiences.items() if not isinstance(v, str) or not v.strip()]
        if invalid_keys:
            print(f"- Warning: removing {len(invalid_keys)} invalid experiences (non-string values): {invalid_keys}")
            for k in invalid_keys:
                del new_experiences[k]
        print("- Num of candidate experiences:", len(new_experiences))

        # consolidate if pool exceeds limit
        if len(new_experiences) > max_pool_size:
            print(f"- Experience pool ({len(new_experiences)}) exceeds limit ({max_pool_size}), consolidating...")
            new_experiences = self._consolidate_experiences(new_experiences, max_pool_size)

        # write to file
        with open(filename, "w") as f:
            json.dump(
                {
                    "operations": all_operations,
                    "response": response,
                    "revision_plan": revision_plan,
                    "new_experiences": new_experiences,
                },
                f,
                indent=2,
            )
        return new_experiences

    @staticmethod
    def _validate_experiences(experiences):
        """Validate that all experience values are non-empty strings."""
        if not isinstance(experiences, dict):
            return False
        for k, v in experiences.items():
            if not isinstance(v, str) or not v.strip():
                return False
        return True

    def _consolidate_experiences(self, experiences, max_pool_size, max_retries=3):
        """When experience pool exceeds limit, call LLM for consolidation."""
        for attempt in range(max_retries):
            try:
                formatted_exps = "\n".join([f"[{k}]. {v}" for k, v in experiences.items()])
                response = self.llm.chat([
                    {"role": "system", "content": EXPERIENCE_CONSOLIDATION_TEMPLATE_SP.format(
                        max_pool_size=max_pool_size)},
                    {"role": "user", "content": EXPERIENCE_CONSOLIDATION_TEMPLATE_UP.format(
                        current_size=len(experiences),
                        max_pool_size=max_pool_size,
                        experiences=formatted_exps)}
                ])
                consolidated = json.loads(response.split("```json")[-1].split("```")[0])
                if not self._validate_experiences(consolidated):
                    print(f"Warning: consolidation attempt {attempt+1} returned invalid format (non-string values), retrying...")
                    continue
                if len(consolidated) <= max_pool_size:
                    print(f"- Consolidated to {len(consolidated)} experiences")
                    return consolidated
            except Exception as e:
                print(f"Warning: consolidation attempt failed: {e}")

        print(f"Warning: consolidation failed after {max_retries} retries, truncating to {max_pool_size}")
        # Fallback: keep first max_pool_size experiences
        items = list(experiences.items())[:max_pool_size]
        return dict(items)

    def _format_exp_and_ops(self, experiences, operations):
        """ Format experiences and operations. """
        if not operations:
            return "No batch operations."
        
        # Format existing experiences and their related operations
        formatted_res = []
        for id, exp in experiences.items():
            curr_str = f"Experience {id}:\nContent: {exp}\n"
            related_ops = [op for op in operations if op.get("id") == id]
            if related_ops:
                curr_str += "Related Operations:\n"
                op_str = []
                for op in related_ops:
                    op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
                op_str = "\n".join(op_str)
                curr_str += op_str
            else:
                curr_str += "No related operations."
            formatted_res.append(curr_str)
        # Format operations without specific IDs
        no_id_ops = [op for op in operations if not op.get("id", None)]
        if no_id_ops:
            curr_str = "Operations without specific Experience ID:\n"
            op_str = []
            for op in no_id_ops:
                op_str.append(f"{json.dumps(op, ensure_ascii=False, indent=2)}")
            op_str = "\n".join(op_str)
            curr_str += op_str
            formatted_res.append(curr_str)

        return "\n\n".join(formatted_res)