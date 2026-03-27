# S(M,T): Scoring AI Endpoints and the Measurement Gap in LLM Routing

**One equation decides which AI endpoint handles a given task.**

S(M,T) scores any endpoint (LLM, agent, script, or tool) against any task by multiplying three things: does it meet hard requirements, how capable is it across multiple dimensions, and is it worth the cost for this specific task. If any dealbreaker fails, the score is zero.

```
S(M,T) = [Gates] × [Compatibility] × [Cost Penalty]
```

That's it. Gates filter out endpoints that can't handle the task. Compatibility measures fit across 16 learned dimensions. Cost adjusts for what the task is worth. The highest-scoring endpoint wins.

**[Download the paper (PDF)](SMT_Router_Paper_v1.1.pdf)**

## Why This Matters

The world is headed toward spending billions on AI tokens every year. Most routing today is manual or nonexistent. Every query gets sent to the same model, regardless of difficulty. A formatting task and a complex reasoning problem get the same $0.06 API call.

S(M,T) routes intelligently. In our tests, it retains 94.35% of best-model quality at half the cost. The savings compound: across a large engineering team, intelligent routing can cut AI spend by 80%+ while matching quality on the tasks that matter.

## Key Results

| Benchmark | Result |
|-----------|--------|
| RouterBench accuracy | 83.63% (vs. 85.51% always-strong, 96.45% oracle) |
| RouterBench 5-shot | 86.76% (beats always-strong 86.66%) |
| RouteLLM AUC | 0.8006 |
| Quality at 50% cost | 94.35% of always-strong |
| Cross-benchmark | One model generalizes across both benchmarks without tuning |

## What's Different

**Heterogeneous endpoints.** Every other router assumes all endpoints are LLMs. S(M,T) scores LLMs, agents, scripts, and tool-use systems through the same equation. A script has no context window. An agent has no per-token cost. The gate layer handles type-specific constraints so the scoring equation doesn't have to.

**The measurement gap.** We tried 14 hand-designed scoring functions on 2.76M benchmark records. All 14 failed. Benchmarks use incomparable scales, don't transfer to deployment, and collapse complex properties into single numbers. This is documented in detail in the paper. It's the finding we think matters most.

**Learned scoring that generalizes.** Instead of hand-designing what "reasoning ability" means as a number, we trained 16 bilinear interaction matrices on 740K routing examples. The model generalizes across benchmarks it wasn't trained for. When we removed an entire data source and retrained, it still worked on the missing source.

**Scaling didn't help.** A 4.6x bigger model got worse on every metric. The bottleneck is data quality, not model capacity.

## Paper

The paper (v1.1, March 2026) covers:

- The S(M,T) equation and why it's designed this way
- The measurement gap: why hand-designed scoring functions fail structurally
- Learned bilinear scoring (3.15M parameters, 740K training samples)
- Cross-benchmark generalization
- Why scaling failed and what that means
- Self-critique: what's genuinely new vs. standard techniques applied to a new problem

Convergence proofs and adversarial analysis are in the appendices.

> v1.1 is a restructured revision. The equation, results, and all experimental data are unchanged from v1.0. See [CHANGELOG.md](CHANGELOG.md) for details. Previous version: [v1.0 release](https://github.com/pranavlakherwal/smt-router/releases/tag/v1.0).

## Code

The `router/` directory contains the core implementation:

| File | Description |
|------|-------------|
| `engine.py` | Main routing engine |
| `scorer.py` | S(M,T) scoring implementation |
| `gates.py` | Multiplicative gate layer |
| `learned_phi.py` | BilinearPhiEngine (16 learned bilinear heads) |
| `train_bilinear.py` | Training script |
| `eval_bilinear.py` | Evaluation on RouterBench and RouteLLM |
| `weight_learner.py` | Online weight learning (Robbins-Monro) |
| `data_pipeline.py` | Unified data ingestion from 5 public datasets |
| `leave_one_out.py` | Leave-one-out generalization analysis |
| `config.py` | Configuration and hyperparameters |
| `model_registry.py` | Model registry (180 aliases to 48 models) |

## Training Data

| Source | Samples | Models | Score Type |
|--------|---------|--------|------------|
| RouterBench | 401,467 | 11 | Accuracy (0/1) |
| RouteLLM | 218,202 | 2 | Win rate (0/1) |
| Arena 55K | 111,712 | 45 | Elo-derived (continuous) |
| MT-Bench | 6,710 | 5 | Rating (1-10, normalized) |
| RewardBench | 2,712 | 31 | Accuracy (0/1) |
| **Total** | **740,803** | **48** | |

## Coming Soon: Shadow Router

An open-source shadow router that monitors your queries, calculates which endpoint it would assign, and logs the result. No interference with your workflow. Think of it like a clinical trial for your AI usage: observe, measure, compare, then decide.

If you're interested in testing it: pranav@pranavlakherwal.com

## Requirements

- Python 3.10+
- PyTorch
- sentence-transformers (all-mpnet-base-v2 for prompt encoding)
- numpy, pandas, scipy

## Citation

```bibtex
@article{lakherwal2026smt,
  title={S(M,T): Scoring AI Endpoints and the Measurement Gap in LLM Routing},
  author={Lakherwal, Pranav},
  year={2026},
  month={March}
}
```

## Looking for arXiv Endorsement

I'm looking for a collaborator with arXiv endorsement in cs.AI, cs.LG, or cs.CL to help submit this paper. If you have endorsement privileges and find this work interesting, please reach out: pranav@pranavlakherwal.com

## License

MIT License. See [LICENSE](LICENSE) for details.

## AI Disclosure

Developed with AI assistance from Anthropic. All research direction, experimental design, and intellectual contributions are the author's own.

## Author

Pranav Lakherwal (pranav@pranavlakherwal.com)
