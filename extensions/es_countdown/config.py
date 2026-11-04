"""Strict configuration for the independent Countdown study."""
import math
from pathlib import Path
import yaml
from dominant_layer_zo.config import merge

DEFAULTS = {
    "model": {"name": "Qwen/Qwen3-8B",
              "revision": "b968826d9c46dd6066d109eabc6255188de91218",
              "dtype": "float16", "attention": "eager"},
    "data": {"train": "data/es-countdown/train.jsonl", "test": "data/es-countdown/test.jsonl",
             "dev": None, "seed": 0, "expected_test_examples": 2000},
    "es": {"population_size": 16, "prompts_per_member": 16, "iterations": 200,
           "sigma": .001, "alpha": .0005, "seed": 0},
    "generation": {"batch_size": 16, "max_input_tokens": 4096, "max_new_tokens": 512,
                   "format_prefix": "<think>"},
    "evaluation": {"dev_every": 0},
    "save_final": False,
    "output_root": "runs/es-countdown",
}
CONDITIONS = {"base": ("none", None), "full": ("full", None),
              "layer6": ("layer", 6), "layer0": ("layer", 0), "layer34": ("layer", 34)}


def load_config(path, overrides=()):
    c = merge(DEFAULTS, yaml.safe_load(Path(path).read_text()))
    for assignment in overrides:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError("Overrides must be section.key=value")
        parts = key.split(".")
        target = c
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config field: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config field: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    if not isinstance(c["model"]["name"], str) or not c["model"]["name"]:
        raise ValueError("Set an explicit model name or local checkpoint")
    if c["model"]["revision"] is not None and not isinstance(c["model"]["revision"], str):
        raise ValueError("Model revision must be a string or null")
    if c["model"]["dtype"] not in ("float16", "float32"):
        raise ValueError("ES raw parameter updates require float16 or float32")
    if c["model"]["attention"] != "eager":
        raise ValueError("This validated runner uses native HF eager attention")
    for section, keys in {"es": ["population_size", "prompts_per_member", "iterations"],
                          "generation": ["batch_size", "max_input_tokens", "max_new_tokens"],
                          "data": ["expected_test_examples"]}.items():
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if c["es"]["population_size"] < 2:
        raise ValueError("ES needs at least two population members")
    for key in ("sigma", "alpha"):
        c["es"][key] = float(c["es"][key])
        if not math.isfinite(c["es"][key]) or c["es"][key] <= 0:
            raise ValueError(f"es.{key} must be finite and positive")
    for section in ("data", "es"):
        if type(c[section]["seed"]) is not int or c[section]["seed"] < 0:
            raise ValueError("Seeds must be nonnegative integers")
    for value in (c["data"]["train"], c["data"]["test"], c["output_root"]):
        if not isinstance(value, str) or not value:
            raise ValueError("Data/output paths must be nonempty strings")
    interval = c["evaluation"]["dev_every"]
    if type(interval) is not int or interval < 0:
        raise ValueError("evaluation.dev_every must be a nonnegative integer")
    dev = c["data"]["dev"]
    if dev is not None and (not isinstance(dev, str) or not dev):
        raise ValueError("data.dev must be a nonempty path or null")
    if (dev is not None) != (interval > 0):
        raise ValueError("Set both data.dev and positive evaluation.dev_every, or disable both")
    if not isinstance(c["generation"]["format_prefix"], str):
        raise ValueError("format_prefix must be a string")
    if type(c["save_final"]) is not bool:
        raise ValueError("save_final must be a boolean")
    return c
