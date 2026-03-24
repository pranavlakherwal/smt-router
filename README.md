# S(M,T): A Unified Scoring Framework for LLM Routing

A scoring framework that decides which AI endpoint (LLM, agent, script, or tool-use system) should handle a given query. One equation scores all endpoint types through the same interface.

## The Equation

```
S(M,T) = [Product_j gate_j(M,T)] * [Sum_i w_i * phi_i(M,T)] * e^{-beta(T) * H}
```

Three parts:
1. **Gate products**: Binary filters (can this endpoint handle the task at all?)
2. **Weighted compatibility**: How well does this endpoint fit this task across K dimensions?
3. **Boltzmann cost**: How expensive is this endpoint relative to task cost sensitivity?

## Key Results

- **RouterBench**: 83.63% routing accuracy across 11 models (vs. 85.51% always-strong baseline, 96.45% oracle)
- **RouteLLM**: AUC 0.8006 across quality-cost thresholds. At 50% strong-model usage, retains 94.35% of always-strong quality
- **BilinearPhiEngine**: 3.15M parameter model with 16 learned bilinear heads, trained on 740K (prompt, model, score) triples from 5 public datasets
- **Measurement gap**: 14 hand-designed phi function designs tested across 2.76M benchmark records, 0/14 produced reliable scores. Learned bilinear forms bypass this entirely.

## Paper

The full paper (31 pages) is in `paper/main.pdf`. It covers:

- Formal definition of S(M,T) with gate, phi, and cost components
- Convergence proofs under Hajek conditions with O(sqrt(KN log N)) regret bounds
- Parameter identifiability and adversarial robustness analysis
- The measurement gap: why hand-designed scoring functions fail on public benchmarks
- BilinearPhiEngine: learned phi functions that bypass the measurement gap
- Leave-one-out generalization analysis across 5 data sources
- Evaluation on RouterBench and RouteLLM benchmarks

## Code

The `router/` directory contains the core implementation:

| File | Description |
|------|-------------|
| `engine.py` | Main routing engine |
| `scorer.py` | S(M,T) scoring implementation |
| `gates.py` | Multiplicative gate layer |
| `phi_evaluators.py` | 16 hand-designed phi function evaluators |
| `learned_phi.py` | BilinearPhiEngine (learned bilinear phi functions) |
| `train_bilinear.py` | Training script for BilinearPhiEngine |
| `eval_bilinear.py` | Evaluation on RouterBench, RouteLLM, internal test set |
| `weight_learner.py` | Online weight learning (Robbins-Monro + SAC entropy decay) |
| `data_pipeline.py` | Unified data ingestion from 5 public routing datasets |
| `leave_one_out.py` | Leave-one-out generalization analysis |
| `config.py` | Configuration and hyperparameters |
| `model_registry.py` | Canonical model registry (180 aliases to 48 models) |
| `models.py` | Data models |
| `registry.py` | Endpoint registry |

## Training Data

The BilinearPhiEngine trains on 5 public datasets unified into a single corpus:

| Source | Samples | Models | Score Type |
|--------|---------|--------|------------|
| RouterBench | 401,467 | 11 | Accuracy (0/1) |
| RouteLLM | 218,202 | 2 | Win rate (0/1) |
| Arena 55K | 111,712 | 45 | Elo-derived (continuous) |
| MT-Bench | 6,710 | 5 | Rating (1-10, normalized) |
| RewardBench | 2,712 | 31 | Accuracy (0/1) |
| **Total** | **740,803** | **48** | |

## Requirements

- Python 3.10+
- PyTorch
- sentence-transformers (all-mpnet-base-v2 for prompt encoding)
- Standard ML stack (numpy, pandas, scipy)

## Citation

If you use this work, please cite:

```bibtex
@article{lakherwal2026smt,
  title={S(M,T): A Unified Scoring Framework for Heterogeneous LLM Routing},
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
