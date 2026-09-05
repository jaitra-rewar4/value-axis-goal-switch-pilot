import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow imports from the repository's common/ folder.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "common"))

from paths import data_file, value_axis
from aime import extract_integer_answer


MODEL_NAME = "Qwen/Qwen3-8B"
LAYER = 21

MATH_SYSTEM = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Put your final integer answer in \\boxed{}."
)

FRAMINGS = {
    "correct": "Do you think your answer is correct? Respond only Yes or No.",
    "incorrect": "Do you think your answer is incorrect? Respond only Yes or No.",
}

PREFILLS = ["Yes", "No"]


def select_examples():
    """Select one correct and one incorrect rollout from the paper's data."""
    rollout_path = data_file("confidence_auroc/rollouts.json")

    with open(rollout_path, "r", encoding="utf-8") as f:
        raw_rollouts = json.load(f)

    selected = {
        "correct_rollout": None,
        "incorrect_rollout": None,
    }

    for rollout in raw_rollouts:
        prediction = extract_integer_answer(rollout["rollout_text"])
        if prediction is None:
            continue

        is_correct = prediction == rollout["answer"]

        if is_correct and selected["correct_rollout"] is None:
            selected["correct_rollout"] = {
                **rollout,
                "predicted_answer": prediction,
                "is_correct": True,
            }

        if not is_correct and selected["incorrect_rollout"] is None:
            selected["incorrect_rollout"] = {
                **rollout,
                "predicted_answer": prediction,
                "is_correct": False,
            }

        if all(example is not None for example in selected.values()):
            break

    if any(example is None for example in selected.values()):
        raise RuntimeError(
            "Could not find both a correct and an incorrect rollout in the dataset."
        )

    return selected


def expected_prefill(is_correct: bool, framing: str) -> str:
    """Return the logically appropriate Yes/No response."""
    if framing == "correct":
        return "Yes" if is_correct else "No"
    return "No" if is_correct else "Yes"


print("Downloading/loading the published Value Axis...")
all_axes = np.load(value_axis())
print("Value Axis shape:", all_axes.shape)

if all_axes.ndim != 2:
    raise RuntimeError(
        f"Expected a 2D Value Axis array, received shape {all_axes.shape}"
    )

if LAYER >= all_axes.shape[0]:
    raise RuntimeError(
        f"Layer {LAYER} is unavailable in axis array with shape {all_axes.shape}"
    )

axis_cpu = torch.from_numpy(all_axes[LAYER]).float()
axis_cpu = axis_cpu / axis_cpu.norm().clamp(min=1e-8)

print("Selecting one correct and one incorrect paper rollout...")
examples = select_examples()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Qwen3-8B in BF16...")
print("The first model download may take several minutes.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

input_device = model.get_input_embeddings().weight.device
axis = axis_cpu.to(input_device)

captured = {}


def layer_hook(_module, _inputs, output):
    """Capture the final-token activation after layer 21."""
    hidden_states = output[0]

    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]

    final_hidden = hidden_states[-1].float()
    final_hidden = final_hidden / final_hidden.norm().clamp(min=1e-8)

    captured["score"] = float(final_hidden @ axis.to(final_hidden.device))


hook_handle = model.model.layers[LAYER].register_forward_hook(layer_hook)

results = []

try:
    for example_name, rollout in examples.items():
        for framing_name, framing_question in FRAMINGS.items():
            for prefill in PREFILLS:
                messages = [
                    {"role": "system", "content": MATH_SYSTEM},
                    {"role": "user", "content": rollout["question"]},
                    {"role": "assistant", "content": rollout["rollout_text"]},
                    {"role": "user", "content": framing_question},
                    {"role": "assistant", "content": prefill},
                ]

                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )

                # Match the repository's balanced-prefill method:
                # truncate immediately after the forced Yes/No response.
                final_occurrence = rendered.rfind(prefill)
                if final_occurrence == -1:
                    raise RuntimeError(
                        f"Could not locate prefill {prefill!r} in rendered prompt."
                    )

                rendered = rendered[: final_occurrence + len(prefill)]

                input_ids = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(input_device)

                captured.clear()

                with torch.inference_mode():
                    model(input_ids=input_ids)

                if "score" not in captured:
                    raise RuntimeError("Layer hook did not capture an activation.")

                result = {
                    "example": example_name,
                    "is_correct": rollout["is_correct"],
                    "true_answer": rollout["answer"],
                    "predicted_answer": rollout["predicted_answer"],
                    "framing": framing_name,
                    "prefill": prefill,
                    "expected_prefill": expected_prefill(
                        rollout["is_correct"], framing_name
                    ),
                    "layer": LAYER,
                    "value_axis_score": captured["score"],
                    "token_count": int(input_ids.shape[-1]),
                }

                results.append(result)

                print(
                    f"{example_name:18s} | "
                    f"{framing_name:9s} | "
                    f"{prefill:3s} | "
                    f"score={captured['score']:+.6f}"
                )

finally:
    hook_handle.remove()


margins = []

for example_name, rollout in examples.items():
    for framing_name in FRAMINGS:
        subset = [
            row
            for row in results
            if row["example"] == example_name
            and row["framing"] == framing_name
        ]

        scores = {
            row["prefill"]: row["value_axis_score"]
            for row in subset
        }

        expected = expected_prefill(rollout["is_correct"], framing_name)
        other = "No" if expected == "Yes" else "Yes"
        margin = scores[expected] - scores[other]

        margins.append(
            {
                "example": example_name,
                "framing": framing_name,
                "expected_prefill": expected,
                "expected_minus_other_margin": margin,
            }
        )


output = {
    "model": MODEL_NAME,
    "layer": LAYER,
    "axis_shape": list(all_axes.shape),
    "examples": examples,
    "scores": results,
    "expected_answer_margins": margins,
}

output_path = Path("smoke_test_value_axis_results.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\nExpected-answer margins:")
print("Positive means the Value Axis favored the logically appropriate Yes/No.")

for row in margins:
    print(
        f"{row['example']:18s} | "
        f"{row['framing']:9s} | "
        f"expected={row['expected_prefill']:3s} | "
        f"margin={row['expected_minus_other_margin']:+.6f}"
    )

print(f"\nSaved complete results to: {output_path}")
