# Before the Answer Flips

## A pilot stress test of Qwen3-8B's Value Axis under competing objectives

This repository contains the code, raw outputs, figures, and report for a short MATS application research project.

The question was simple: when an instruction pushes Qwen3-8B away from factual correctness and toward a learned grader rule, does the published Value Axis detect that movement before the final answer flips?

## Read the report

- [Full report](REPORT.md)
- [Compact item-level results](pilot_v2_compact.csv)
- [Revised pilot raw data](pilot_v2_raw.json)
- [Revised pilot summary](pilot_v2_summary.json)
- [Agent-use disclosure](AGENT_USE.md)

## Result

The revised four-item pilot ranked the grader's hidden target label first on all four items, although the absolute rule-diagnostic evidence was weak. Qwen chose the correct answer under the factual objective in 4/4 items and still chose it under the grader objective in 4/4 items. Switching to the grader objective nevertheless moved the rewarded proxy upward relative to the correct answer on every item, by +3.19 logits on average. The Value Axis moved slightly toward the proxy at the answer-label token (+0.00164), but moved slightly in the opposite direction on an identical-suffix position control (-0.00037). I therefore do not treat the pilot as evidence that the Value Axis tracks the induced objective change.

## Why the negative result matters

The experiment produced a plausible positive-looking answer-token result. The stricter suffix control did not agree. That disagreement is the main finding. It shows why an internal direction should not be interpreted from one convenient token position, especially when the behavioral manipulation itself did not flip the output.

## Repository map

- `smoke_test_value_axis.py`: verifies model, layer hook, axis loading, and projection.
- `pilot_conflicting_objectives.py`: first pilot, kept because its failed manipulation motivated the redesign.
- `pilot_goal_switch_v2.py`: balanced active-goal pilot used in the report.
- `analysis/recompute_headline_results.py`: recomputes headline numbers and verifies structural controls.
- `pilot_v1_raw.json` and `pilot_v1_summary.json`: first-pilot outputs.
- `pilot_v2_raw.json`, `pilot_v2_summary.json`, and `pilot_v2_compact.csv`: revised-pilot outputs.
- `fig1_behavior_goal_switch.png`, `fig2_value_label_goal_switch.png`, and `fig3_value_suffix_goal_switch.png`: figures generated from the revised pilot.
- `REPORT.md`: full write-up.
- `AGENT_USE.md`: LLM and coding-agent disclosure.

## Reproduction

The scripts run from the root of the official Value Axis repository.

```bash
git clone https://github.com/nickjiang2378/value-axis.git upstream-value-axis
cd upstream-value-axis
git checkout 44ad182f1e43902858748d8d4b93835c58ddfe4c
uv sync
source .venv/bin/activate
```

Copy the three experiment scripts from this repository into that repository root and run:

```bash
python smoke_test_value_axis.py
python pilot_conflicting_objectives.py
python pilot_goal_switch_v2.py
```

To recompute the released summary without a GPU:

```bash
python analysis/recompute_headline_results.py
```

## Exact setup

- Model: `Qwen/Qwen3-8B`
- Layer: 21
- Value Axis shape: 37 x 4,096
- Precision: BF16
- GPU: NVIDIA RTX 6000 Ada Generation
- Upstream commit: `44ad182f1e43902858748d8d4b93835c58ddfe4c`
- Goal prompt length: 102 tokens in both conditions
- Final sample: 4 paired items

## Scope

This is a small exploratory pilot, not a general evaluation of the Value Axis. It does not show reward hacking, a persistent hidden goal, or that the Value Axis generally succeeds or fails. The claim is narrower: behavior shifted consistently toward the grader proxy, while the internal measurement depended on where it was taken.

## References

1. Jiang, N., Kauvar, I., and Lindsey, J. (2026). *The Value Axis: Language Models Encode Whether They're on the Right Track*. arXiv:2606.17056.
2. Pan, A., Jones, E., Jagadeesan, M., and Steinhardt, J. (2024). *Feedback Loops With Language Models Drive In-Context Reward Hacking*. arXiv:2402.06627.
