# LLM and agent use

I used GPT-5.6 Sol through ChatGPT and coding-agent workflows throughout this project. Because I was new to much of the mechanistic-interpretability tooling, I used it to help understand the Value Axis paper and repository, set up and debug the remote GPU environment, scaffold and debug experiment code, think through alternative explanations, produce initial plots, and critique the write-up.

I did not use an LLM judge for any reported result. The measurements in the report come from saved Qwen3-8B logits and layer-21 activations. The repository includes the experiment scripts, raw JSON outputs, compact results, figures, and the upstream Value Axis commit used for the run.

I tried to prioritize checks that could change the conclusion rather than treating an agent-generated summary as evidence. The first pilot was kept after its manipulation check failed instead of being hidden or scaled. The revised experiment used matched factual and grader conditions, balanced demonstrations, single-token answer labels, equal-length goal prompts, and an identical-suffix control. The headline quantities can also be recomputed from the released raw outputs with `analysis/recompute_headline_results.py`.

The control that mattered most was the identical-suffix measurement. The answer-label Value Axis result initially looked mildly positive, but the matched-suffix result slightly reversed, so I narrowed the conclusion rather than interpreting the nicer-looking result as confirmation. I would be more confident in the arithmetic and saved measurements than in the broader interpretation of this four-item pilot. I also did not independently audit every line of the upstream Value Axis implementation, and the experiment is too small to support broad generalization.
