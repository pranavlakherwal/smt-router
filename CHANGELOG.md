# Changelog

All notable changes to S(M,T) will be documented in this file.

## [v1.3] - 2026-06-02

### Shadow Routing Production Update + Rust SGLang Gateway Port (skeleton)

Two months of observations from a shadow-routing deployment of S(M,T) on a personal-twin daemon, plus a skeleton Rust port of the policy targeting SGLang's Model Gateway. Also adds a worked example of S(M,T) routing across the Grok model family.

**What changed:**

- **New `docs/GROK_FAMILY_ROUTING.md`:** worked example of S(M,T) applied to a publicly-priced model family (the Grok lineup), with six query types walked through cell-by-cell and an aggregate cost-vs-quality table. Demonstrates that the equation handles intra-family routing across model + `reasoning_effort` jointly — a dimension the existing open routers do not exploit. Weighted result: ~65% cost saved at -3.5pp quality on a representative mix.
- **New branch `grok-spacexai-Rust-routingpolicy`** carries a draft Rust crate (`rust-sgl-policy-smt/`) that mirrors the eventual upstream layout for `sgl-model-gateway/src/policies/s_mt/`. Implements the `RoutingPolicy` trait with `select_single`, `select_pair`, and (most importantly) `on_request_complete` — the outcome hook that no built-in SGLang policy currently uses. Robbins-Monro weight updates live there. The crate compiles in isolation against stubbed SGLang types; trait shapes will be pinned against current `sgl-project/sglang` HEAD before any upstream PR.
- **Production deployment observations (Q1-Q2 2026, 345 routing decisions logged).** Under online Robbins-Monro updates from outcome signals, two features dominate the converged weights: `phi_1` (relevance) ≈ 0.474 and `phi_8` (composition headroom) ≈ 0.526. The other fourteen `phi` features zeroed out. Consistent with the v1.1 measurement-gap framing: routing is feature-bound, not capacity-bound.
- **Gate-pruning rate observed at ~44%** on the production sample (77-decision JSONL slice analysed). The eligibility filter is doing most of the practical work; the score discriminates among the surviving set. Consistent with the v1.2 endpoint-typology intuition that gates carry type-specific constraints.

**What did not change:**

- The equation, all formal definitions, training data, model architecture.
- v1.1 experimental results (83.63% RouterBench, AUC 0.8006 RouteLLM, 94.35% quality at 50% cost).
- v1.2 endpoint typology forward references.
- The canonical Python implementation in `router/`.

## [v1.2] - 2026-03-27

### Endpoint Typology

Identifies six structural properties of endpoints that the current S(M,T) equation cannot express. These properties distinguish computation patterns (scripts, LLM calls, tool-augmented LLMs, agents) that the equation currently treats as points in the same capability space.

**What changed:**

- **New Section 8.4: "Endpoint Typology: What the Equation Cannot Express."** Names the core problem: the equation scores *what* an endpoint can do, not *how* it computes. Documents six missing dimensions: autonomy, compositionality, state management, determinism, scope of action, and recursion depth.
- **Four candidate parameters proposed:** autonomy α ∈ [0,1], composition depth d ∈ ℕ, state function σ(t), determinism δ ∈ [0,1]. Each could enter as new phi functions or new gates.
- **Future Directions updated** to reference the typology as a source of new scoring functions and gates.
- Cites existing agent orchestration work (MoMA, DAAO) already in bibliography.

**What did not change:**

- All experimental results, the equation, training data, and implementation.
- Existing sections are unchanged except for a one-sentence forward reference in Future Directions.

## [v1.1] - 2026-03-27

### Revised Edition

This is a substantial revision of the original paper, restructured for clarity and conciseness.

**What changed:**

- **Paper restructured from 14 pages to ~9 pages.** Convergence proofs, adversarial analysis, seed weight derivation, and entropy scheduling moved to appendices. Main body focuses on the equation, the measurement gap, and learned results.
- **New title:** "S(M,T): Scoring AI Endpoints and the Measurement Gap in LLM Routing"
- **New abstract** following Goal, Problem, Solution, Result, Impact structure.
- **New introduction** with concrete opening example, ubiquitous routing thesis, explicit hypothesis, and four clearly scoped contributions.
- **Added design reasoning section (Section 2.1).** Explains *why* these three terms, *why* multiply, and *why* cost tolerance belongs to the task.
- **Self-critique section added (Section 8.2).** Documents systematic falsification of novelty claims. Acknowledges that S(M,T) maps to conditional logit, bilinear forms are factorization machines, and convergence proofs are standard. The paper's posture shifted from "novel techniques" to "novel application, novel empirical finding, novel direction."
- **Goodhart/Campbell/Manheim framing** added to measurement gap root causes.
- **Deeper engagement with GraphRouter** as the closest architectural comparison.
- **Shadow routing** added as future direction (coming soon, open source).
- **Bibliography expanded** with 11 new citations: McFadden (1974), Train (2009), Fishburn (1967), Rendle (2010), Koren (2009), Jacobs (1991), Jordan (1994), Vaswani (2017), Goodhart (1975), Campbell (1979), Manheim (2019).
- **Cross-benchmark generalization** elevated as a primary result with explicit interpretation.
- **Scaling failure** reframed as measurement gap confirmation.
- **Heterogeneous routing** positioned as the primary differentiator with specific examples.

**What did not change:**

- All experimental results (83.63% RouterBench, AUC 0.8006 RouteLLM, 94.35% quality at 50% cost).
- The S(M,T) equation and all formal definitions.
- Training data, model architecture, and implementation details.
- All convergence proofs and adversarial analysis (moved to appendix, not removed).

## [v1.0] - 2026-03-25

### Initial Release

- Original paper: "S(M,T): A Unified Scoring Framework for LLM Routing with Provable Convergence and the Measurement Gap"
- 14 sections across ~14 pages
- Full convergence proofs in main body
- Tagged as `v1.0` in git
