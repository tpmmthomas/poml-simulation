"""Deterministic benchmark sampling with saved, portable token manifests."""

import random

DATASETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "hellaswag": ("Rowan/hellaswag", "default", "validation"),
    "piqa": ("regisss/piqa", "default", "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "resisting_correction": ("pminervini/inverse-scaling", "resisting-correction", "data"),
}


def benchmark_examples(tokenizer, task, *, count=100, seed=42, prompt_length=32):
    """Sample public benchmark rows; preserve exact scored tokens in the result."""
    from datasets import load_dataset

    if task not in DATASETS or count < 1:
        raise ValueError("unknown dataset or nonpositive sample count")
    name, config, split = DATASETS[task]
    dataset = load_dataset(name, config, split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    examples = []

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    for index in indices:
        row = dataset[index]
        choices, answer, target = [], None, []
        if task == "wikitext2":
            tokens = encode(row["text"])
            if len(tokens) <= prompt_length:
                continue
            context, target = tokens[:prompt_length], tokens[prompt_length : prompt_length + 32]
        elif task == "lambada":
            text, last = row["text"].rsplit(" ", 1)
            context, target = encode(text)[-prompt_length:], encode(" " + last)
        elif task == "resisting_correction":
            # Keep the full adversarial instruction; truncating its beginning
            # would change the benchmark's intended behavior.
            context = encode(row["prompt"])
            if len(context) + 33 > 1024:
                continue
        else:
            text = (
                row["ctx"]
                if task == "hellaswag"
                else row["goal"]
                if task == "piqa"
                else row["question"]
            )
            context = encode(text)[-prompt_length:]
            endings = (
                row["endings"]
                if task == "hellaswag"
                else [row["sol1"], row["sol2"]]
                if task == "piqa"
                else row["choices"]["text"]
            )
            choices = [encode(" " + value.strip()) for value in endings]
            answer = (
                int(row["label"])
                if task != "arc_easy"
                else row["choices"]["label"].index(row["answerKey"])
            )
        if not context or (choices and not all(choices)):
            continue
        examples.append(
            {
                "id": f"{task}:{index}",
                "task": task,
                "context": context,
                "target": target,
                "choices": choices,
                "answer": answer,
                "source": {"dataset": name, "config": config, "split": split},
            }
        )
        if len(examples) == count:
            return examples
    raise ValueError(f"only {len(examples)} usable examples for {task}, requested {count}")
