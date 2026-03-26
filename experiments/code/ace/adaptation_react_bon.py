import copy
import json
import os
import random
import uuid
from typing import Any

from appworld.common.utils import chunk_and_return
from appworld.evaluator import evaluate_task
from appworld_experiments.code.ace.adaptation_agent import StarAgent
from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent
from appworld_experiments.code.ace.evaluation_react import SimplifiedReActAgent


@StarAgent.register("ace_adaptation_react_bon")
class SimplifiedReActBestOfNStarAgent(SimplifiedReActStarAgent):
    def __init__(
        self,
        generator_model_config: dict,
        reflector_model_config: dict,
        curator_model_config: dict,
        generator_prompt_file_path: str | None = None,
        reflector_prompt_file_path: str | None = None,
        curator_prompt_file_path: str | None = None,
        initial_playbook_file_path: str | None = None,
        trained_playbook_file_path: str | None = None,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        appworld_config: dict | None = None,
        logger_config: dict | None = None,
        max_steps: int = 40,
        max_cost_overall: float = 3000,
        max_cost_per_task: float = 10,
        log_lm_calls: bool = False,
        use_reflector: bool = True,
        num_retries: int = 5,
        use_gt_code: bool = False,
        num_curator_samples: int = 4,
        playbook_delta_eval_mode: str = "current_task_pass_cases",
        playbook_delta_eval_num_tasks: int = 5,
        playbook_delta_eval_seed: int | None = None,
        playbook_delta_eval_exclude_current_task: bool = True,
    ):
        self._generator_model_config = copy.deepcopy(generator_model_config)
        self._generator_prompt_file_path = generator_prompt_file_path
        self._appworld_config = copy.deepcopy(appworld_config or {})
        self._logger_config = copy.deepcopy(logger_config or {})
        self._evaluation_max_steps = max_steps
        self._evaluation_max_cost_overall = max_cost_overall
        self._evaluation_max_cost_per_task = max_cost_per_task
        self._evaluation_log_lm_calls = log_lm_calls

        self.num_curator_samples = max(1, num_curator_samples)
        self.playbook_delta_eval_mode = playbook_delta_eval_mode
        self.playbook_delta_eval_num_tasks = max(1, playbook_delta_eval_num_tasks)
        self.playbook_delta_eval_seed = playbook_delta_eval_seed
        self.playbook_delta_eval_exclude_current_task = playbook_delta_eval_exclude_current_task
        self.training_task_ids: list[str] = []

        super().__init__(
            generator_prompt_file_path=generator_prompt_file_path,
            reflector_prompt_file_path=reflector_prompt_file_path,
            curator_prompt_file_path=curator_prompt_file_path,
            initial_playbook_file_path=initial_playbook_file_path,
            trained_playbook_file_path=trained_playbook_file_path,
            ignore_multiple_calls=ignore_multiple_calls,
            max_prompt_length=max_prompt_length,
            max_output_length=max_output_length,
            generator_model_config=generator_model_config,
            reflector_model_config=reflector_model_config,
            curator_model_config=curator_model_config,
            appworld_config=appworld_config,
            logger_config=logger_config,
            max_steps=max_steps,
            max_cost_overall=max_cost_overall,
            max_cost_per_task=max_cost_per_task,
            log_lm_calls=log_lm_calls,
            use_reflector=use_reflector,
            num_retries=num_retries,
            use_gt_code=use_gt_code,
        )

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        num_tasks = len(task_ids)
        num_processes = min(num_processes, num_tasks)
        self.training_task_ids = chunk_and_return(
            task_ids, num_chunks=num_processes, chunk_index=process_index
        )
        super().solve_tasks(
            task_ids=task_ids,
            experiment_name=experiment_name,
            num_processes=num_processes,
            process_index=process_index,
        )

    def _apply_candidate_operations(
        self, base_playbook: str, operations: list[dict]
    ) -> tuple[str, int]:
        return self._apply_operations_to_playbook(
            operations, playbook=base_playbook, next_global_id=self.next_global_id
        )

    def _sample_eval_task_ids(self, current_task_id: str) -> list[str]:
        if self.playbook_delta_eval_mode == "current_task_pass_cases":
            return [current_task_id]

        if self.playbook_delta_eval_mode != "sampled_train_pass_rate":
            raise ValueError(
                f"Unknown playbook delta eval mode: {self.playbook_delta_eval_mode}"
            )

        candidate_task_ids = list(self.training_task_ids)
        if self.playbook_delta_eval_exclude_current_task and len(candidate_task_ids) > 1:
            candidate_task_ids = [task_id for task_id in candidate_task_ids if task_id != current_task_id]
        if not candidate_task_ids:
            candidate_task_ids = [current_task_id]

        rng_seed = self.playbook_delta_eval_seed
        if rng_seed is not None:
            rng_seed += self.current_task_index
        rng = random.Random(rng_seed)

        sample_size = min(self.playbook_delta_eval_num_tasks, len(candidate_task_ids))
        if sample_size >= len(candidate_task_ids):
            return candidate_task_ids
        return rng.sample(candidate_task_ids, sample_size)

    def _get_eval_artifact_root_path(self) -> str:
        if not self.trained_playbook_file_path:
            return os.path.join(os.getcwd(), "tasks-dynamic-eval")
        experiment_root = os.path.dirname(
            os.path.dirname(os.path.dirname(self.trained_playbook_file_path))
        )
        return os.path.join(experiment_root, "tasks-dynamic-eval")

    def _get_eval_experiment_prefix(self) -> str:
        if not self.trained_playbook_file_path:
            return "tasks-dynamic-eval"
        normalized_path = os.path.normpath(self.trained_playbook_file_path)
        outputs_marker = f"{os.sep}outputs{os.sep}"
        if outputs_marker not in normalized_path:
            return "tasks-dynamic-eval"
        relative_playbook_path = normalized_path.split(outputs_marker, 1)[1]
        relative_parts = relative_playbook_path.split(os.sep)
        if len(relative_parts) < 4:
            return "tasks-dynamic-eval"
        base_experiment_parts = relative_parts[:-3]
        return os.path.join(*base_experiment_parts, "tasks-dynamic-eval")

    def _prepare_eval_artifact_paths(
        self, eval_task_id: str, eval_run_name: str
    ) -> tuple[str, str]:
        current_task_id = getattr(getattr(self, "world", None), "task", None)
        current_task_id = getattr(current_task_id, "id", "unknown_task")

        eval_artifact_directory = os.path.join(
            self._get_eval_artifact_root_path(),
            current_task_id,
            eval_run_name,
        )
        os.makedirs(eval_artifact_directory, exist_ok=True)

        playbook_path = os.path.join(
            eval_artifact_directory, f"{eval_task_id}_playbook.txt"
        )
        experiment_name = os.path.join(
            self._get_eval_experiment_prefix(),
            current_task_id,
            eval_run_name,
        )
        return playbook_path, experiment_name

    def _run_eval_task_with_playbook(
        self, task_id: str, playbook: str, eval_run_name: str
    ) -> tuple[float, Any]:
        logger_config = copy.deepcopy(self._logger_config)
        logger_config["verbose"] = False

        temp_playbook_path, eval_experiment_name = self._prepare_eval_artifact_paths(
            eval_task_id=task_id, eval_run_name=eval_run_name
        )
        with open(temp_playbook_path, "w") as temp_playbook_file:
            temp_playbook_file.write(playbook)

        eval_agent = SimplifiedReActAgent(
            generator_model_config=copy.deepcopy(self._generator_model_config),
            generator_prompt_file_path=self._generator_prompt_file_path,
            trained_playbook_file_path=temp_playbook_path,
            ignore_multiple_calls=self.ignore_multiple_calls,
            max_prompt_length=self.max_prompt_length,
            max_output_length=self.max_output_length,
            appworld_config=copy.deepcopy(self._appworld_config),
            logger_config=logger_config,
            max_steps=self._evaluation_max_steps,
            max_cost_overall=self._evaluation_max_cost_overall,
            max_cost_per_task=self._evaluation_max_cost_per_task,
            log_lm_calls=self._evaluation_log_lm_calls,
        )
        eval_agent.solve_task(task_id, eval_experiment_name)
        test_tracker, _ = evaluate_task(task_id, eval_experiment_name)

        if self.playbook_delta_eval_mode == "current_task_pass_cases":
            return float(test_tracker.pass_count), test_tracker
        return float(test_tracker.pass_count) / max(test_tracker.num_tests, 1), test_tracker

    def evaluate_playbook_delta(
        self,
        candidate_playbook: str,
        eval_task_ids: list[str] | None = None,
        eval_run_name: str | None = None,
    ) -> dict[str, Any]:
        if eval_task_ids is None:
            current_task_id = self.world.task.id
            eval_task_ids = self._sample_eval_task_ids(current_task_id)
        eval_run_name = eval_run_name or f"eval_{uuid.uuid4().hex[:8]}"
        per_task_scores: list[float] = []
        pass_counts: list[int] = []
        pass_rates: list[float] = []

        for eval_task_id in eval_task_ids:
            score, test_tracker = self._run_eval_task_with_playbook(
                eval_task_id, candidate_playbook, eval_run_name=eval_run_name
            )
            per_task_scores.append(score)
            pass_counts.append(test_tracker.pass_count)
            pass_rates.append(float(test_tracker.pass_count) / max(test_tracker.num_tests, 1))

        return {
            "score": sum(per_task_scores) / max(len(per_task_scores), 1),
            "task_ids": eval_task_ids,
            "per_task_scores": per_task_scores,
            "pass_counts": pass_counts,
            "pass_rates": pass_rates,
            "eval_run_name": eval_run_name,
        }

    def curator_call(self):
        reasoning_text = None
        if self.use_reflector:
            reasoning_text = self.reflector_call()

        content = self._build_curator_input(reasoning_text)
        self.last_curator_input = content
        self.curation_messages = [{"role": "user", "content": content}]

        base_seed = self.curator_model.generation_kwargs.get("seed")
        eval_task_ids = self._sample_eval_task_ids(self.world.task.id)
        candidate_records: list[dict[str, Any]] = []

        for sample_idx in range(self.num_curator_samples):
            generation_kwargs = {}
            if base_seed is not None:
                generation_kwargs["seed"] = base_seed + sample_idx

            curator_raw = self.curator_model.generate(
                messages=self.curation_messages, **generation_kwargs
            )
            curator_response = curator_raw.get("content", "")
            candidate_record: dict[str, Any] = {
                "sample_idx": sample_idx,
                "response": curator_response,
                "score": float("-inf"),
                "operations": [],
                "eval_details": None,
                "error": None,
            }

            # import pdb; pdb.set_trace()
            # try:
            operations = self._parse_curator_operations(curator_response)
            candidate_playbook, _ = self._apply_candidate_operations(self.playbook, operations)
            eval_run_name = f"candidate_{sample_idx}_{uuid.uuid4().hex[:8]}"
            eval_details = self.evaluate_playbook_delta(
                candidate_playbook,
                eval_task_ids=eval_task_ids,
                eval_run_name=eval_run_name,
            )
            candidate_record["operations"] = operations
            candidate_record["score"] = eval_details["score"]
            candidate_record["eval_details"] = eval_details
            # except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            #     candidate_record["error"] = str(error)
            # except Exception as error:
            #     candidate_record["error"] = str(error)

            candidate_records.append(candidate_record)

        score_list = [candidate["score"] for candidate in candidate_records]
        print(
            f"📊 Playbook delta scores for task {self.world.task.id}: {score_list}"
        )
        for candidate in candidate_records:
            eval_details = candidate["eval_details"] or {}
            print(
                "   "
                f"candidate={candidate['sample_idx']} "
                f"score={candidate['score']} "
                f"per_task_scores={eval_details.get('per_task_scores', [])} "
                f"pass_counts={eval_details.get('pass_counts', [])} "
                f"pass_rates={eval_details.get('pass_rates', [])} "
                f"eval_run_name={eval_details.get('eval_run_name')}"
            )

        best_candidate = max(candidate_records, key=lambda candidate: candidate["score"])
        selected_response = best_candidate["response"]

        if best_candidate["operations"]:
            self.playbook, self.next_global_id = self._apply_operations_to_playbook(
                best_candidate["operations"],
                playbook=self.playbook,
                next_global_id=self.next_global_id,
            )
            print(
                "✅ Selected curator candidate "
                f"{best_candidate['sample_idx']} with score {best_candidate['score']}"
            )
        else:
            print("⏭️  No valid curator candidate found; keeping existing playbook")

        self._persist_playbook()

        self.last_curator_output = json.dumps(
            {
                "selected_sample_idx": best_candidate["sample_idx"],
                "selected_score": best_candidate["score"],
                "selected_eval_details": best_candidate["eval_details"],
                "shared_eval_task_ids": eval_task_ids,
                "candidates": [
                    {
                        "sample_idx": candidate["sample_idx"],
                        "score": candidate["score"],
                        "eval_details": candidate["eval_details"],
                        "error": candidate["error"],
                    }
                    for candidate in candidate_records
                ],
                "selected_response": selected_response,
            },
            ensure_ascii=True,
        )

        self._show_curator_response(selected_response)
