"""Run one independent Countdown condition; no test-set tuning or checkpoint resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, set_seed
from dominant_layer_zo.methods.scope import select_parameters
from dominant_layer_zo.runtime import atomic_json, environment, run_lock
from .config import CONDITIONS, load_config
from .countdown import load_examples, check_disjoint, score_response
from .optimizer import EvolutionStrategies


class ValidateScores(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Validate generation scores before computing rewards.
        # HF processors may intentionally mask some tokens with negative infinity.
        invalid = torch.isnan(scores).any() | torch.isposinf(scores).any()
        if invalid or not torch.isfinite(scores).any(dim=-1).all():
            raise FloatingPointError("Invalid Countdown generation scores")
        return scores


class Evaluator:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.generated_nonpad_tokens = 0
        self.examples_processed = 0
        self.generation_calls = 0

    @torch.inference_mode()
    def evaluate(self, rows):
        self.model.eval()
        config = self.config
        results = []
        device = next(self.model.parameters()).device
        for start in range(0, len(rows), config["batch_size"]):
            batch = rows[start:start + config["batch_size"]]
            inputs = self.tokenizer([row["prompt"] for row in batch], return_tensors="pt",
                                    padding=True, truncation=False, add_special_tokens=True)
            inputs.pop("token_type_ids", None)
            length = inputs["input_ids"].shape[1]
            if length > config["max_input_tokens"]:
                raise ValueError("Countdown prompt exceeds max_input_tokens; refusing silent truncation")
            limit = getattr(self.model.config, "max_position_embeddings", None)
            if limit and length + config["max_new_tokens"] > limit:
                raise ValueError("Prompt plus generation budget exceeds model context")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=config["max_new_tokens"], use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                logits_processor=[ValidateScores()],
            )
            completion = output[:, length:]
            responses = self.tokenizer.batch_decode(completion, skip_special_tokens=True)
            self.generated_nonpad_tokens += int((completion != self.tokenizer.pad_token_id).sum())
            self.examples_processed += len(batch)
            self.generation_calls += 1
            for row, response in zip(batch, responses):
                response = response.strip()
                results.append({"response": response, **score_response(
                    response, row["numbers"], row["target"], config["format_prefix"])})
        n = len(results)
        return {"accuracy": sum(row["correct"] for row in results) / n,
                "mean_reward": sum(row["reward"] for row in results) / n,
                "examples": n}, results


def source_signature():
    from dominant_layer_zo.methods import scope
    paths = sorted(Path(__file__).parent.glob("*.py")) + [Path(scope.__file__)]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(c, condition, device="cuda"):
    scope, layer = CONDITIONS[condition]
    output = Path(c["output_root"]) / condition
    with run_lock(output) as out:
        started = time.monotonic()
        record = {"status": "running", "condition": condition, "scope": scope, "layer": layer,
                  "config": c, "implementation_signature": source_signature(),
                  "environment": environment(),
                  "protocol": {"reference": "arxiv:2509.24372v1",
                               "backend": "serial_huggingface", "objective": "countdown_reward",
                               "restoration": "exact_parameter_dtype_cpu_snapshot",
                               "update_accumulator": "float32_per_tensor",
                               "noise": "gaussian_independent_named_tensor_seeds",
                               "reward_normalization": "float32_population_std_plus_1e-8",
                               "selection": "terminal_iteration", "historical_reproduction": False}}
        atomic_json(out / "config.json", c)
        atomic_json(out / "run.json", record)
        try:
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise ValueError("Countdown runner requires one process")
            if torch.device(device).type == "cuda" and torch.cuda.device_count() != 1:
                raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
            test, test_meta = load_examples(c["data"]["test"])
            if len(test) != c["data"]["expected_test_examples"]:
                raise ValueError("Test split size differs from expected_test_examples")
            train, train_meta = ([], None)
            if condition != "base":
                train, train_meta = load_examples(c["data"]["train"])
                check_disjoint(train, test)
                if c["es"]["prompts_per_member"] > len(train):
                    raise ValueError("Not enough distinct training prompts")
            dev, dev_meta = ([], None)
            if c["data"]["dev"] is not None:
                dev, dev_meta = load_examples(c["data"]["dev"])
                check_disjoint(train, dev)
                check_disjoint(dev, test)
            record["data"] = {"train": train_meta, "dev": dev_meta, "test": test_meta}
            set_seed(c["data"]["seed"])
            print(f"{condition}: loading {c['model']['name']} ({c['model']['dtype']})", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                c["model"]["name"], revision=c["model"]["revision"],
                torch_dtype=getattr(torch, c["model"]["dtype"]),
                attn_implementation=c["model"]["attention"], low_cpu_mem_usage=True,
            ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(c["model"]["name"], revision=c["model"]["revision"])
            if tokenizer.eos_token_id is None:
                raise ValueError("Countdown requires an EOS token")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            record["model_revision"] = getattr(model.config, "_commit_hash", None)
            record["effective_attention"] = model.config._attn_implementation
            evaluator = Evaluator(model, tokenizer, c["generation"])
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            completed = 0

            def evaluate_dev(iteration):
                dev_started = time.monotonic()
                metrics, predictions = evaluator.evaluate(dev)
                observation = {"iteration": iteration, **metrics,
                               "seconds": time.monotonic() - dev_started}
                with (out / "dev_history.jsonl").open("a") as history:
                    history.write(json.dumps(observation, allow_nan=False) + "\n")
                atomic_json(out / "dev_predictions.json", predictions)
                record["latest_dev"] = observation
                atomic_json(out / "run.json", record)
                print(f"{condition} fixed dev at iteration {iteration}: "
                      f"reward={metrics['mean_reward']:.6f} accuracy={metrics['accuracy']:.4f}",
                      flush=True)

            if dev:
                evaluate_dev(0)
            if condition != "base":
                parameters = select_parameters(model, scope, layer)
                record["trainable_parameters"] = sum(p.numel() for _, p in parameters)
                record["cpu_snapshot_bytes"] = sum(p.numel() * p.element_size() for _, p in parameters)
                atomic_json(out / "run.json", record)
                opts = c["es"]
                optimizer = EvolutionStrategies(parameters, population_size=opts["population_size"],
                                                sigma=opts["sigma"], alpha=opts["alpha"], seed=opts["seed"])
                sampler = random.Random(c["data"]["seed"])
                with (out / "history.jsonl").open("w") as history, \
                     (out / "population.jsonl").open("w") as population:
                    for iteration in range(opts["iterations"]):
                        indices = sampler.sample(range(len(train)), opts["prompts_per_member"])
                        batch = [train[i] for i in indices]
                        step_start = time.monotonic()
                        member = 0

                        def reward():
                            nonlocal member
                            member_started = time.monotonic()
                            tokens_before = evaluator.generated_nonpad_tokens
                            metrics, _ = evaluator.evaluate(batch)
                            member += 1
                            progress = {"iteration": iteration + 1, "member": member,
                                        **metrics, "seconds": time.monotonic() - member_started,
                                        "generated_nonpad_tokens": evaluator.generated_nonpad_tokens - tokens_before}
                            population.write(json.dumps(progress, allow_nan=False) + "\n")
                            population.flush()
                            print(f"{condition} iteration {iteration + 1}/{opts['iterations']} "
                                  f"member {member}/{opts['population_size']}: "
                                  f"reward={metrics['mean_reward']:.6f} "
                                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
                            return metrics["mean_reward"]

                        stats = optimizer.step(reward)
                        if torch.device(device).type == "cuda":
                            torch.cuda.synchronize()
                        completed = iteration + 1
                        stats.update(iteration=completed, training_indices=indices,
                                     seconds=time.monotonic() - step_start,
                                     cumulative_generated_nonpad_tokens=evaluator.generated_nonpad_tokens)
                        history.write(json.dumps(stats, allow_nan=False) + "\n")
                        history.flush()
                        record["completed_iterations"] = completed
                        atomic_json(out / "run.json", record)
                        print(f"{condition} iteration {completed}/{opts['iterations']}: "
                              f"reward={stats['mean_reward']:.6f} "
                              f"std={stats['reward_std']:.6f} "
                              f"update_l2={stats['actual_update_l2']:.6g} "
                              f"changed={stats['changed_parameter_entries']}", flush=True)
                        if dev and (completed % c["evaluation"]["dev_every"] == 0
                                    or completed == opts["iterations"]):
                            evaluate_dev(completed)
            print(f"{condition}: final evaluation on {len(test)} examples", flush=True)
            metrics, predictions = evaluator.evaluate(test)
            metrics["accuracy_percent"] = 100 * metrics["accuracy"]
            atomic_json(out / "downstream_eval.json", metrics)
            atomic_json(out / "predictions.json", predictions)
            if c["save_final"] and condition != "base":
                model.save_pretrained(out / "final")
                tokenizer.save_pretrained(out / "final")
            record.update(status="complete", completed_iterations=completed, metrics=metrics,
                          objective_evaluations=completed * c["es"]["population_size"],
                          examples_processed=evaluator.examples_processed,
                          generation_calls=evaluator.generation_calls,
                          generated_nonpad_tokens=evaluator.generated_nonpad_tokens,
                          seconds_including_loading=time.monotonic() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None)
            atomic_json(out / "run.json", record)
            return metrics
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(out / "run.json", record)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS), required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.dry_run:
        print(json.dumps({"condition": args.condition, "config": config}, indent=2))
    else:
        print(json.dumps(run(config, args.condition, args.device), indent=2))


if __name__ == "__main__":
    main()
