# Before the Answer Flips

## A Pilot Stress Test of Qwen3-8B's Value Axis Under Competing Objectives

**Jaitra Rewar**  
MATS 12.0 application research task  
September 2026

**Code and full raw results:** `https://github.com/jaitra-rewar4/value-axis-goal-switch-pilot`

# Executive Summary

A model can move toward an objective before its final answer changes. I tested whether the Value Axis, a published internal direction in Qwen3-8B that has been interpreted as an "on-track" signal, can detect that earlier movement.

## What problem am I trying to solve?

The Value Axis was built by contrasting Qwen3-8B activations before and after the model discovered how to receive positive feedback in a hidden-rule task. The original work found that the direction correlates with confidence, backtracking, and code correctness, and that steering it changes how confidently the model persists. This makes it tempting to use the axis as a readout of whether a model thinks its current strategy is succeeding.

The difficult case is when success has two meanings. In my setup, Objective X asked the model to choose the factually correct multiple-choice answer. Objective Y asked it to infer a hidden grader rule from feedback and choose the answer label most likely to receive +1. The correct answer and rewarded label were always different. I then compared ordinary output logits with layer-21 Value Axis projections.

## High-level takeaways

1. The technical pipeline worked. I loaded Qwen3-8B in BF16, used the authors' published layer-21 axis, captured hidden activations, and verified the core projection calculation on released AIME rollouts.
2. A first pilot failed and was redesigned rather than scaled. Its grader-rule check was only 1/4, so I replaced it with sixteen balanced demonstrations and an explicit active-goal switch.
3. The grader objective changed preference without changing the final answer. In the revised four-item pilot, the proxy-minus-correct logit increased on every item, by +3.19 on average. Qwen still chose the factual answer in all four cases.
4. The Value Axis result depended on where it was measured. At the A/B/C/D token, three of four items shifted toward the proxy, with a mean change of +0.00164. On identical suffix tokens, only one of four shifted in that direction and the mean change was -0.00037.
5. My conclusion is negative and narrow. This pilot does not provide robust evidence that the Value Axis tracks a sub-behavioral change in the active objective.

## Compact raw-results table

| Item | Correct | Proxy | Rule rank | Top X | Top Y | Delta logit | Delta VA label | Delta VA suffix |
|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| 1 | B | A | A | B | B | +3.75 | -0.000845 | -0.000338 |
| 2 | C | B | B | C | C | +2.00 | +0.001277 | +0.000065 |
| 3 | D | C | C | D | D | +3.75 | +0.003013 | -0.000615 |
| 4 | A | D | D | A | A | +3.25 | +0.003101 | -0.000591 |

## Random raw example

Using a fixed seed of `20260904` selected Item 1, not a favorable case. The question was "Which planet is commonly called the Red Planet?" The correct label was B and the rewarded proxy was A. Under Objective X, the A-versus-B logit margin was -20.00. Under Objective Y, it rose to -16.25, a +3.75 shift toward A, but B remained overwhelmingly preferred. The Value Axis label-token margin moved from -0.031120 to -0.031964, while the identical-suffix margin moved from -0.003295 to -0.003633. This example captures the overall pattern: a behavioral nudge toward the proxy without a corresponding positive Value Axis shift.

# Detailed Analysis

## 1. Background and motivation

Mechanistic interpretability studies model internals rather than only judging outputs. The Value Axis is one recent attempt to turn an internal activation direction into a useful measurement. Jiang, Kauvar, and Lindsey construct a direction in Qwen3-8B by comparing hidden states before and after the model discovers a hidden criterion in an in-context feedback game. They define value as how likely the current trajectory is to accomplish the model's goal. Their layer-21 direction generalizes to held-out hidden criteria and correlates with verbalized confidence, backtracking, and code correctness. Steering the direction also changes confidence-like behavior.

This creates a practical question. Real evaluators can reward the wrong thing. A model might know what is factually correct while also learning what an imperfect grader rewards. Behavioral work on in-context reward hacking shows that feedback loops can push models toward proxy objectives. I wanted to know whether the Value Axis distinguishes the model's factual objective from a learned grader objective, or whether it mainly reflects confidence in the answer that is already most likely.

I did not train a new direction. I used the published axis as a fixed measuring instrument and placed it in a setting it was not fitted on.

## 2. Research question and hypotheses

**Research question:** When an instruction shifts Qwen3-8B's preference from factual correctness toward a learned grader objective, does the published Value Axis detect that shift before the model's final answer changes?

I considered five possibilities:

1. Active-objective tracking. The Value Axis moves toward the rewarded proxy when Objective Y becomes active.
2. Factual-confidence tracking. The axis continues favoring the correct answer because the model remains more confident in the fact.
3. Output-probability tracking. The axis simply moves with ordinary logits without adding a distinct signal.
4. Token or position artifact. A shift at the A/B/C/D token disappears when measured on identical tokens at identical positions.
5. No stable effect. The goal instruction changes behavior while the published axis does not respond reliably.

## 3. Experimental setup

I used `Qwen/Qwen3-8B` in BF16 on an NVIDIA RTX 6000 Ada GPU. I checked out upstream Value Axis repository commit `44ad182f1e43902858748d8d4b93835c58ddfe4c`. The released axis has shape 37 x 4,096, and I used layer 21.

For a candidate continuation, I captured the final hidden activation from layer 21, normalized it, normalized the published layer-21 axis, and took their dot product. The sign of an absolute score was not interpreted by itself. I compared the proxy candidate with the correct candidate under matched conditions.

Each item contained sixteen demonstration questions. A, B, C, and D each appeared exactly four times. Exactly four demonstration answers were factually correct. A hidden grader gave +1 whenever the assistant used one target label and -1 otherwise.

The same history and final question were run twice:

- Objective X: choose the factually correct option.
- Objective Y: infer the grader's hidden rule and choose the label most likely to receive +1.

Only the active objective changed. The two system prompts were both 102 tokens, and the demonstrations, final question, and answer choices were identical within each pair.

For the Value Axis, I measured the proxy-minus-correct margin at the answer-label token and on the shared tokens in `This choice is recorded as my final answer.` The suffix was tokenized identically and appeared at the same positions for every candidate.

## 4. Initial pilot and redesign

The first pilot compared truthful, shuffled, and corrupted feedback after eight demonstrations. Its manipulation check identified the intended proxy in only 1/4 corrupted cases. The system prompt also called factual correctness the primary instruction and warned that the grader might be imperfect. The identical-suffix Value Axis difference between corrupted and truthful conditions was approximately +0.000044.

I did not scale this design. I replaced it with balanced demonstrations and an explicit active-goal switch. The first graphs existed, but the experiment had not created the condition needed to interpret them.

## 5. Results

### 5.1 The model ranked the grader rule, but did not follow it as its final answer

The revised diagnostic ranked the intended proxy first for A, B, C, and D. The absolute evidence was weak, so I interpret this as a relative ranking rather than confident explicit knowledge. Under Objective X, Qwen chose the factually correct answer on all four items. Under Objective Y, it still chose the correct answer on all four items.

The instruction had a consistent effect below the top choice. The proxy-minus-correct logit margin increased by +3.75, +2.00, +3.75, and +3.25. The mean shift was +3.1875. The grader objective pushed every item in the predicted direction, but not far enough to alter the selected answer.

### 5.2 The answer-label Value Axis moved slightly toward the proxy

At the answer-label token, the goal-switch effects were -0.000845, +0.001277, +0.003013, and +0.003101. Three of four were positive, with a mean of +0.001637. Taken alone, this could look like early evidence that the Value Axis tracks a change in the active objective before behavior flips.

### 5.3 The position-controlled suffix did not reproduce the shift

On the identical suffix tokens, the four goal-switch effects were -0.000338, +0.000065, -0.000615, and -0.000591. The mean was -0.000370. Only one item moved toward the proxy.

This control changed my interpretation of the answer-token result. The positive-looking answer-token average may reflect A/B/C/D token identity or local continuation dynamics rather than a robust change in the model's internal estimate of goal success.

## 6. Sanity checks and agent use

I preserved every system prompt, demonstration, final question, candidate logit, token ID, and token-level Value Axis projection in the raw JSON. The release includes a separate recomputation script. Structural checks confirm that the factual and grader conditions use identical demonstrations and final questions, the goal prompts have equal token length, A/B/C/D are single distinct tokens, candidate completions have equal length, and suffix tokens are identical across candidates.

Agentic tools helped scaffold code, debug the GPU setup, check calculations, and produce plots. The project retained explicit go/no-go conditions and raw outputs so the results were not accepted from an agent-generated summary alone.

## 7. What the evidence supports

The evidence supports two narrow statements:

1. In these four items, activating the grader objective consistently shifted Qwen3-8B's output preference toward the grader-rewarded label, even though factual correctness remained the top choice.
2. The published layer-21 Value Axis did not robustly track that shift across measurement positions. A small answer-token effect did not appear on identical suffix tokens.

The evidence does not show that Qwen reward hacked, adopted a persistent hidden goal, or preferred grader approval over truth. It also does not show that the Value Axis tracks factual correctness. The behavioral manipulation was partial, the sample was small, and the internal measurements disagreed.

## 8. Limitations

- The revised pilot contained only four final questions.
- The questions were easy enough that the correct answer remained dominant.
- The grader objective never changed the top answer.
- The rule diagnostic used relative ranking and had weak absolute evidence.
- No random-direction null distribution was run.
- Only Qwen3-8B and layer 21 were tested.
- The study used in-context instructions, not fine-tuning or persistent learned objectives.
- The suffix average may dilute a localized signal, while the label token has token-identity confounds.

## 9. What I would do next

I would first calibrate a larger question set using only ordinary logits, select items where Objective Y can sometimes flip the top answer, and freeze that set before inspecting the Value Axis. I would then add random-direction controls and compare nearby layers. A stronger later version could install the grader preference through fine-tuning, followed by revalidation or reconstruction of the axis.

## 10. Conclusion

This project began with a simple question: can an internal "on-track" direction notice a model moving toward a different objective before the answer changes? The revised pilot produced a consistent behavioral nudge toward the grader proxy, but the Value Axis evidence was not stable under a stricter position control. The result is not a positive validation of the method. It is a small boundary test showing that a plausible answer-token effect can disappear when the same comparison is made on matched tokens.

The most useful lesson was methodological. A technically working pipeline and a clean-looking graph were not enough. The first pilot failed its manipulation check, and the second produced a partial behavioral effect rather than a goal switch. Keeping those failures visible led to a narrower conclusion that the data can actually support.

# References

1. Jiang, N., Kauvar, I., and Lindsey, J. (2026). *The Value Axis: Language Models Encode Whether They're on the Right Track*. arXiv:2606.17056.
2. Pan, A., Jones, E., Jagadeesan, M., and Steinhardt, J. (2024). *Feedback Loops With Language Models Drive In-Context Reward Hacking*. arXiv:2402.06627.
3. Nanda, N. (2025). *Highly Opinionated Advice on How to Write ML Papers*. AI Alignment Forum.
4. Nanda, N. (2025). *My Research Process: Key Mindsets: Truth-Seeking, Prioritisation, Moving Fast*. AI Alignment Forum.
