# Changelog

All notable changes to S(M,T) will be documented in this file.

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
