from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
raw = json.loads((ROOT / "results" / "pilot_v2_raw.json").read_text())
summary = json.loads((ROOT / "results" / "pilot_v2_summary.json").read_text())

logit, label, suffix = [], [], []
rule, factual, grader = [], [], []
for r in raw:
    c = r["goal_switch_comparison"]
    logit.append(c["grader_minus_factual_behavior_margin"])
    label.append(c["grader_minus_factual_value_margin_label_token"])
    suffix.append(c["grader_minus_factual_value_margin_common_suffix"])
    rule.append(r["grader_rule_knowledge"]["matches_proxy"])
    factual.append(r["goals"]["factual_goal"]["derived"]["behavior_satisfies_active_goal"])
    grader.append(r["goals"]["grader_goal"]["derived"]["behavior_satisfies_active_goal"])

observed = {
    "rule_knowledge_accuracy": float(np.mean(rule)),
    "factual_goal_behavior_accuracy": float(np.mean(factual)),
    "grader_goal_behavior_accuracy": float(np.mean(grader)),
    "mean_logit_goal_switch_effect": float(np.mean(logit)),
    "mean_value_label_goal_switch_effect": float(np.mean(label)),
    "mean_value_suffix_goal_switch_effect": float(np.mean(suffix)),
}
expected = {
    "rule_knowledge_accuracy": summary["rule_knowledge_accuracy"],
    "factual_goal_behavior_accuracy": summary["active_goal_behavior_accuracy"]["factual_goal"],
    "grader_goal_behavior_accuracy": summary["active_goal_behavior_accuracy"]["grader_goal"],
    "mean_logit_goal_switch_effect": summary["means"]["behavior_logit_margin"]["mean_goal_switch_effect"],
    "mean_value_label_goal_switch_effect": summary["means"]["value_margin_label_token"]["mean_goal_switch_effect"],
    "mean_value_suffix_goal_switch_effect": summary["means"]["value_margin_common_suffix"]["mean_goal_switch_effect"],
}
for key in observed:
    assert np.isclose(observed[key], expected[key]), (key, observed[key], expected[key])

for r in raw:
    x = r["goals"]["factual_goal"]
    y = r["goals"]["grader_goal"]
    assert x["demonstrations"] == y["demonstrations"]
    assert x["final_messages"][1:] == y["final_messages"][1:]
    for goal in [x, y]:
        candidates = goal["candidate_value_scores"]
        lengths = {len(candidates[label]["content_token_ids"]) for label in "ABCD"}
        suffixes = {tuple(candidates[label]["content_token_ids"][1:]) for label in "ABCD"}
        assert len(lengths) == 1
        assert len(suffixes) == 1

print(json.dumps(observed, indent=2))
print("All headline values and structural controls match the raw data.")
