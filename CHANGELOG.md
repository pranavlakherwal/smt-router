# Changelog

All notable changes to S(M,T) will be documented in this file.

## [paper 1.5] - 2026-11-07

### The bar as a gate, the quality gap named, the November 2026 measurement in the paper

Paper v1.5 resolves SMT-ERR-004 and SMT-ERR-005 of `paper/ERRATA_QUEUE.md`. The original
claims stay in place; each one now carries what it rests on.

**Paper, three edits:**
- `paper/sections/03_framework.tex`: the measured accuracy bar is the sixth gate (J=6), with the
  selection rule when a bar is set: the cheapest endpoint, by measured cost per task, among those
  that cleared the bar on 100 tasks and again on 100 fresh tasks. The score is still computed and
  does not choose. The maths of the section is unchanged.
- `paper/sections/07d_production_validation.tex`: a new subsection, "What This Section Did Not
  Measure". The 94.68% cost cut across 1,580 shadow decisions had no measure of answer quality
  beside it. On 6 November 2026 the router's list held 373 models: 35 with their own leaderboard
  row, 56 with a sibling's row, 282 guessed from price. The free endpoint picked on 244 of the 400
  November tasks had been withdrawn.
- `paper/sections/07f_november_2026_measurement.tex` (new, a subsection of the production
  section): 56 models in 66 settings, 100 tasks on each of four kinds of job, 100 fresh tasks for
  the cheapest hits. Table 20 sets the score's pick beside the cheapest setting that cleared 90,
  95 and 98 twice, from Tables 1, 3 and 6 of `docs/ROUTE_OPTIONS_2026-11.md` at commit 8b5fd5f.
  SMT-ERR-005 in one paragraph: the exponent is computed from confidence, cost enters through five
  price tiers, 61 of 66 settings sit in one tier, so the score picks the most accurate setting.
- The abstract points at both new subsections. Title block: Version 1.5, 7 November 2026.
- `SMT_Router_Paper_v1.5.pdf` at the root (v1.4 kept); `paper/main.pdf` recompiled with tectonic.
  SHA256 97a91b9ca794db27399137fd215ad4faaaeff802c114f98837e2766a9bc07961, stamped with
  OpenTimestamps on 7 November 2026 (`registration/SMT_Router_Paper_v1.5.pdf.ots`, pending until
  the calendar's Bitcoin transaction confirms).

**README:**
- The 80% line is back, with its arithmetic and date: 20% of queries to the strong model and 80%
  to the cheap one kept 89.02% of the strong model's quality on the paper's RouteLLM table
  (Section 8.5.2, Table 9; in the paper since 25 March 2026; confirmed run on 7 November 2026).
- Links the v1.5 PDF; "paper v1.4" wording updated where it described the current paper.
- The recommended call, `python -m router route --kind maths --bar 95 "<task>"`, with the note
  that 95 is the bar we use ourselves by default, decided 7 November 2026.

**What it does not change:** the code, the data and the tests of 0.2.0. Published Medium and
Reddit copies of the paper are not yet updated.

## [code 0.2.0] - 2026-11-06

### The engine runs as published, and it carries a measured table

The paper is unchanged at v1.4. This release is code, data and a research note.

**Why:** on 4 November 2026 we ran the published engine as a stranger would. It could not find
its seed weights or its phi rules, because `.gitignore` had kept the whole `data/` folder out of
the repository. It printed two warnings and routed anyway. It also routed an empty query. On 5
and 6 November we measured 56 models in 66 settings on 400 tasks and found that the capability
numbers our own router used were mostly leaderboard rows and guesses from price.

**What changed:**
- `data/measured_capability.json`: right answers out of 100 per setting and kind of job, with
  measured cost, list price and date. `data/outcomes_2026-11-06.json` holds right or wrong for
  every setting on every task. `data/suspect_keys_2026-11-06.json` lists the answer keys the
  strongest settings disagree with.
- `router/measured.py`: the rule. The option at a bar is the cheapest setting that reached the
  bar on 100 tasks and again on 100 fresh ones.
- `router/gates.py`: a sixth gate, `measured_bar`. With `Task(bar=90, kind="maths")` an endpoint
  passes only if it cleared the bar twice, and the cheapest that passes is selected.
- `router/phi_evaluators.py`: phi_4 reads the measured pass rate. Every score names its source.
- `python -m router options | route | table | report`, and `examples/route_one.py`.
- Seed weights and phi rules now ship. A missing file is an error, not a silent fallback. An
  empty query and a misnamed field are errors. Nothing is written unless `SMT_ROUTER_STATE_DIR`
  is set.
- The default registry is the 66 measured settings. The paper's hand-typed endpoints of mid 2025
  load only with `EndpointRegistry(include_paper_examples=True)`.
- 34 tests, `pyproject.toml`, `uv.lock`, `ruff.toml`, and a manual-only CI workflow.
- `docs/ROUTE_OPTIONS_2026-11.md`: the method, eight tables and the limits.
- `paper/ERRATA_QUEUE.md`: SMT-ERR-004 and SMT-ERR-005, both OPEN.

**What it does not change:** the learned bilinear scorer has not been run on the November tasks.
Nothing here has been checked by an outside party.

## [v1.4] - 2026-06-11

### Discrete-choice attribution correction + full May-31 section restructure

This release corrects an attribution failure in the original paper. Our own
nightly adversarial verification (Ramanujan engine) found, one day after the
March publication, that the S(M,T) scoring form is mathematically a constrained
conditional logit: the gates are a choice-set restriction and the cost term is
a linear utility component (McFadden, 1974; Train, 2009), with Mixture-of-Experts
gating (Jacobs et al., 1991; Shazeer et al., 2017) as ML prior art. The finding
went unrouted for 2.5 months because no errata loop existed. Both problems are
fixed.

**What changed:**
- Related work gains a Discrete Choice Theory paragraph with full attribution.
- Novelty claims repositioned in the abstract, introduction, and conclusion:
  the FORM of the S(M,T) score is not new; the parameterization for heterogeneous endpoint
  routing (local agents as routing destinations), the identifiability,
  convergence, and regret analysis of that parameterization, the measurement-gap
  result, and the 61-day production deployment are the contributions.
- paper/ now carries the full May-31 restructured sections (07a-07e: production
  validation, endpoint profiler) that the June-2 push missed.
- paper/MCFADDEN_RESOLUTION.md: the exact algebra of what is and is not
  equivalent, and the timeline of the miss.
- paper/ERRATA_QUEUE.md: a standing public errata queue, now fed nightly by the
  adversarial verification loop. Two items remain OPEN (decision-count snapshot
  reconciliation; a stale novelty-survival sentence in Section 5) and will be
  drained in the arXiv v2 pass.
- Recompiled main.pdf; root SMT_Router_Paper_v1.4.pdf added.

## [v1.3] - 2026-06-02

### Shadow Routing Production Update + Rust SGLang Gateway Port (skeleton)

Two months of observations from a shadow-routing deployment of S(M,T) on a personal-twin daemon, plus a skeleton Rust port of the policy targeting SGLang's Model Gateway. Also adds a worked example of S(M,T) routing across the Grok model family.

**What changed:**

- **New `docs/GROK_FAMILY_ROUTING.md`:** worked example of S(M,T) applied to a publicly-priced model family (the Grok lineup), with six query types walked through cell-by-cell and an aggregate cost-vs-quality table. Demonstrates that the S(M,T) score handles intra-family routing across model + `reasoning_effort` jointly, a dimension the existing open routers do not exploit. Weighted result: ~65% cost saved at -3.5pp quality on a representative mix.
- **New branch `grok-spacexai-Rust-routingpolicy`** carries a draft Rust crate (`rust-sgl-policy-smt/`) that mirrors the eventual upstream layout for `sgl-model-gateway/src/policies/s_mt/`. Implements the `RoutingPolicy` trait with `select_single`, `select_pair`, and (most importantly) `on_request_complete`, the outcome hook that no built-in SGLang policy currently uses. Robbins-Monro weight updates live there. The crate compiles in isolation against stubbed SGLang types; trait shapes will be pinned against current `sgl-project/sglang` HEAD before any upstream PR.
- **Production deployment observations (Q1-Q2 2026, 345 routing decisions logged).** Under online Robbins-Monro updates from outcome signals, two features dominate the converged weights: `phi_1` (relevance) ≈ 0.474 and `phi_8` (composition headroom) ≈ 0.526. The other fourteen `phi` features zeroed out. Consistent with the v1.1 measurement-gap framing: routing is feature-bound, not capacity-bound.
- **Gate-pruning rate observed at ~44%** on the production sample (77-decision JSONL slice analysed). The eligibility filter is doing most of the practical work; the score discriminates among the surviving set. Consistent with the v1.2 endpoint-typology intuition that gates carry type-specific constraints.

**What did not change:**

- The S(M,T) score, all formal definitions, training data, model architecture.
- v1.1 experimental results (83.63% RouterBench, AUC 0.8006 RouteLLM, 94.35% quality at 50% cost).
- v1.2 endpoint typology forward references.
- The canonical Python implementation in `router/`.

## [v1.2] - 2026-03-27

### Endpoint Typology

Identifies six structural properties of endpoints that the current S(M,T) score cannot express. These properties distinguish computation patterns (scripts, LLM calls, tool-augmented LLMs, agents) that the score currently treats as points in the same capability space.

**What changed:**

- **New Section 8.4 on endpoint typology: what the scoring form cannot express.** Names the core problem: the score says *what* an endpoint can do, not *how* it computes. Documents six missing dimensions: autonomy, compositionality, state management, determinism, scope of action, and recursion depth.
- **Four candidate parameters proposed:** autonomy α ∈ [0,1], composition depth d ∈ ℕ, state function σ(t), determinism δ ∈ [0,1]. Each could enter as new phi functions or new gates.
- **Future Directions updated** to reference the typology as a source of new scoring functions and gates.
- Cites existing agent orchestration work (MoMA, DAAO) already in bibliography.

**What did not change:**

- All experimental results, the S(M,T) score, training data, and implementation.
- Existing sections are unchanged except for a one-sentence forward reference in Future Directions.

## [v1.1] - 2026-03-27

### Revised Edition

This is a substantial revision of the original paper, restructured for clarity and conciseness.

**What changed:**

- **Paper restructured from 14 pages to ~9 pages.** Convergence proofs, adversarial analysis, seed weight derivation, and entropy scheduling moved to appendices. Main body focuses on the S(M,T) score, the measurement gap, and learned results.
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
