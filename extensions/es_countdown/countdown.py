"""Countdown data and grading, with arithmetic parsed instead of Python eval."""
import ast
from collections import Counter
import hashlib
import json
import math
import operator
from pathlib import Path
import re


def arithmetic(expression):
    if not expression or len(expression) > 512 or not re.fullmatch(r"[0-9+\-*/() ]+", expression):
        raise ValueError("Invalid Countdown expression")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub,
              ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        raise ValueError("Only integer literals and + - * / are supported")

    result = float(visit(tree.body))
    if not math.isfinite(result):
        raise ValueError("Nonfinite arithmetic")
    return result


def score_response(response, numbers, target, format_prefix="<think>"):
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    correct = False
    if matches:
        expr = matches[-1]
        try:
            used = [int(x) for x in re.findall(r"\d+", expr)]
            correct = Counter(used) == Counter(numbers) and abs(arithmetic(expr) - target) < 1e-5
        except (ValueError, SyntaxError, ArithmeticError, RecursionError):
            pass
    formatted = format_prefix + response
    if re.fullmatch(r"<think>.*?</think>\n<answer>.*?</answer>", formatted, re.DOTALL):
        formatting = 1.0
    else:
        formatting = .1 * bool(re.search(r"<think>.*?</think>", formatted, re.DOTALL))
        formatting += .5 * bool(re.search(r"<answer>.*?</answer>", formatted, re.DOTALL))
    return {"correct": bool(correct), "format_reward": formatting,
            "reward": float(correct) + .1 * formatting}


def validate_row(row):
    prompt = row.get("prompt", row.get("context"))
    numbers, target = row.get("numbers"), row.get("target")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Each example needs a nonempty prompt/context")
    if not isinstance(numbers, list) or not numbers or any(type(n) is not int or n < 0 for n in numbers):
        raise ValueError("numbers must be a nonempty list of nonnegative integers")
    if isinstance(target, str):
        target = float(target)
        if math.isfinite(target) and target.is_integer():
            target = int(target)
    if type(target) not in (int, float) or not math.isfinite(target):
        raise ValueError("target must be a finite number")
    return {"prompt": prompt, "numbers": numbers, "target": target}


def load_examples(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        values = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    else:
        values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty JSON array or JSONL dataset")
    rows = [validate_row(row) for row in values]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, {"path": str(path), "examples": len(rows), "sha256": digest}


def check_disjoint(train, test):
    def key(row):
        return tuple(sorted(row["numbers"])), row["target"]
    if {key(row) for row in train} & {key(row) for row in test}:
        raise ValueError("Countdown train/test contain overlapping puzzles")
