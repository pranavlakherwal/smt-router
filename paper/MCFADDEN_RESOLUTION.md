# McFadden 1974 Resolution

**Date**: 2026-06-11
**Paper**: S(M,T): A Unified Scoring Framework for LLM Routing (published 2026-03-24 to GitHub, Medium, Reddit)
**Status**: RESOLVED-IN-TEX (this session). See `ERRATA_QUEUE.md` entry SMT-ERR-001.

---

## 1. The Finding, Verbatim

The nightly Ramanujan adversarial engine first surfaced the McFadden equivalence on **2026-03-25, one day after publication**, and re-found it in **58 nightly reports** through 2026-06-10 without any change reaching the tex.

**First occurrence** (`memory/reports/ramanujan/2026-03-25.md`, item 12, VERIFIED, confidence 98%, DeepSeek-R1):

> - S(M,T) is structurally equivalent to a constrained conditional logit model with a specific utility parameterization
> - The gate layer corresponds exactly to choice set constraints in discrete choice theory
> - The Boltzmann term e^(-beta*H) is mathematically identical to the logit kernel

**Latest occurrence** (`memory/reports/ramanujan/2026-06-10.md`, item 12, CITATION AUDIT, VERIFIED, confidence 95%, Qwen3-Coder-480B):

> - S(M,T) exhibits mathematical equivalence to constrained conditional logit models with Boltzmann-weighted choice probabilities
> - The gate layer corresponds to choice set constraints C_n ⊂ J well-established in discrete choice literature
> - The phi functions represent domain-specific systematic utilities V_i(x) = β^T * x_i from McFadden's framework

**Novelty attack** (`memory/reports/ramanujan/2026-06-10.md`, item 5, counter_evidence, claim REFUTED, confidence 98%, DeepSeek-R1):

> - The S(M,T) equation is algebraically equivalent to a constrained multinomial logit (MNL) model after log-transformation, with domain-specific feature engineering.
> - The multiplicative structure decomposes into additive utility components under softmax normalization, matching McFadden's 1974 formulation exactly.
> - The 'gate * compat * cost' factorization provides no mathematical innovation beyond standard discrete choice theory foundations.

**Sharpest statement of the verdict** (`memory/reports/ramanujan/2026-05-23.md`, item 17, claim REFUTED, confidence 90%, DeepSeek-R1):

> - The multiplicative structure gate*compat*cost is simply exp(U(M,T)), a standard form in discrete choice theory dating back to McFadden (1974).
> - The novelty claim for the equation form is therefore invalid; the true contribution lies in the application to LLM routing and the specific design choices within that domain.

Cited prior art across the reports: McFadden (1974) "Conditional Logit Analysis of Qualitative Choice Behavior"; Ben-Akiva and Lerman (1985); Train (2009) "Discrete Choice Methods with Simulation"; Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"; Shazeer et al. (2017) "Outrageously Large Neural Networks".

---

## 2. The Exact Algebra

The published equation, with J = 5 binary gates, K = 16 compatibility functions, w on the simplex, beta(T) and H(M) nonnegative vectors:

```
S(M,T) = [prod_j g_j(M,T)] * [sum_i w_i phi_i(M,T)] * exp(-beta(T) . H(M))
```

Take the log (defined whenever all gates pass and the weighted sum is positive):

```
log S(M,T) = sum_j log g_j(M,T)  +  log( sum_i w_i phi_i(M,T) )  -  beta(T)^T H(M)
```

McFadden's conditional logit (1974): random utility U_i = V_i + eps_i with iid extreme-value errors gives choice probability P(i | x) = exp(V_i) / sum_{j in C} exp(V_j), where V_i = beta^T x_i is linear in attributes and C is the choice set of available alternatives.

### What maps EXACTLY to McFadden

**Term 1, the gate product = choice-set restriction.** Each g_j is binary, so sum_j log g_j is 0 if every gate passes and -infinity if any fails. Under selection proportional to S (equivalently, softmax over log S), a -infinity utility means probability exactly zero. This is identical to restricting the choice set C(T) subset of M, the standard availability constraint of discrete choice analysis. Exact correspondence, no caveats.

**Term 3, the cost term = a linear-in-parameters utility component.** -beta(T)^T H(M) is linear in the endpoint attribute vector H(M) with task-conditional coefficients beta(T). Coefficients that vary with decision-maker (here, task) characteristics are standard interaction terms in conditional logit. This is exactly McFadden's V = beta^T x form. Exact correspondence.

### What does NOT map: the log-sum compatibility term

**Term 2 is not an additive-in-features MNL utility.** A standard conditional logit has V linear in parameters and features. The middle term contributes log(sum_i w_i phi_i(M,T)): the log of a weighted sum, nonlinear in both w and phi. By the identity

```
log( sum_i w_i phi_i ) = log( sum_i exp( log w_i + log phi_i ) )
```

it has the inclusive-value (log-sum-exp) functional form of nested logit (Train 2009, ch. 4: I_k = ln sum_j exp(V_j / lambda_k)) with unit nesting coefficient and pseudo-alternative utilities log w_i + log phi_i. But it is not literally a nested-logit inclusive value: the phi_i are bounded feature values in [0,1], not utilities of real sub-alternatives, and the w_i live on a probability simplex rather than being free parameters. So under softmax-of-log-S, the system is a conditional logit over a constrained choice set with **one nonstandard regressor** of inclusive-value form, coefficient fixed at 1.

One consequence: the Ramanujan finding's strongest phrasing ("decomposes into additive utility components ... matching McFadden's 1974 formulation exactly") is itself a slight overclaim. The decomposition is additive across the three components, but the middle component is not additive in the phi features. The 95 to 98% confidence verdict on form-level equivalence stands; the word "exactly" applies to terms 1 and 3 only.

### Decision rules

- **Boltzmann sampling** (P(M|T) proportional to S(M,T)): this IS a conditional logit over the constrained choice set with systematic utility U(M,T) = log(sum_i w_i phi_i) - beta(T)^T H(M).
- **Deterministic argmax**: argmax_M S = argmax_M log S, the MAP alternative of the same logit, i.e. its zero-temperature limit. The equivalence carries over.

One difference of kind, recorded for completeness: McFadden's model is an econometric model estimated by maximum likelihood from observed choices. S(M,T) is an engineered scoring function whose parameters come from literature priors and Robbins-Monro online learning. The equivalence is of functional form, not of estimation methodology. This does not rescue the equation-form novelty claim, because the form is what the paper claimed.

### What is NOT subsumed by McFadden 1974

1. **The log-sum compatibility parameterization** as a design choice (a different functional form from the linear MNL utility; it changes what the identifiability and learning analysis must handle).
2. **The identifiability, convergence, and regret theorems as stated for THIS parameterization** (Theorems 1 to 5 in sections 04): Robbins-Monro updates on simplex-projected w, Hajek cooling, the contextual-bandit regret reduction. The audit separately flags that Theorem 3 (weight identifiability) is a standard full-rank linear regression condition; that is a distinct, lower-severity attribution issue, not the McFadden issue.
3. **The heterogeneous-endpoint instantiation**: typed gates and phi/cost definitions over LLMs, agents, scripts, and tool systems within one scoring interface.
4. **The empirical results**: the measurement gap (0/14 grounding designs), the 61-day production deployment (94.7% cost reduction, 25.4% agent routing), and the learned bilinear results. These are empirical findings, not structural claims, and are untouched by the equivalence.

ML prior art reaches the same structure from the gating side: mixture-of-experts gating (Jacobs et al. 1991) and sparsely-gated MoE (Shazeer et al. 2017) are softmax choice models with hard choice-set restriction.

---

## 3. The Integrity Issue, Stated Plainly

The paper claimed equation-form contribution ("a unified scoring equation with formal guarantees ... The contribution is the specific three-part multiplicative composition") without citing 50-year-old discrete choice theory that contains the form. McFadden published the conditional logit in 1974 and received the 2000 Nobel prize partly for it. The paper's own related-work section cited McFadden only for the Boltzmann cost layer, not for the full-equation equivalence.

Worse than the missing citation is the broken loop. The monitoring system did its job: it found the equivalence on 2026-03-25, one day after publication, at 98% confidence, and re-found it in 58 nightly reports across 2.5 months. Section 05 of the paper even claims "All six devil's-advocate challenges were refuted ... The paper's novelty claims survive adversarial attack," which later runs directly contradict (2026-05-23 and 2026-06-10 both REFUTE the equation-form novelty claim at 90 to 98%). Nothing routed the finding back into the tex. The failure is the missing errata loop: adversarial findings accumulated in reports nobody acted on, while the published artifact stayed wrong. `ERRATA_QUEUE.md` (created this session) is the structural fix; the Ramanujan pipeline should write to it and a human or agent should drain it.

---

## 4. The Resolution Applied (this session, 2026-06-11)

1. **`sections/08_related_work.tex`**: added a dedicated "Discrete Choice Theory" paragraph stating the equivalence honestly: under log transform and softmax, S(M,T) is a constrained conditional logit (McFadden 1974) with a log-sum compatibility regressor; gates are choice-set restriction; the cost term is a linear utility component; the form is not new. Cites mcfadden1974conditional, train2009discrete, jacobs1991adaptive, shazeer2017outrageously. Also amended contribution item (1) in "What S(M,T) Adds" to stop claiming the composition form and claim the parameterization instead.
2. **`sections/01_intro.tex`**: added the lineage statement after the equation: no novelty is claimed for the equation form.
3. **`main.tex` (abstract)**: added one sentence positioning the form as a constrained conditional logit and the contribution as parameterization, analysis, and empirical program.
4. **`sections/10_conclusion.tex`**: same repositioning in the concluding summary.
5. **`bibliography.bib`**: verified mcfadden1974conditional, train2009discrete, jacobs1991adaptive, shazeer2017outrageously all already present; no additions needed.
6. **`ERRATA_QUEUE.md`**: created with normalized schema; McFadden entry RESOLVED-IN-TEX; decision-count drift and the stale Section 5 novelty-survival claim logged as OPEN for the arXiv v2 pass.
7. PDF recompiled with tectonic.

Not done here (out of scope, needs owner action): updating the already-published Medium/Reddit/GitHub copies, and the arXiv v2 pass that drains the OPEN errata entries.

---

## 5. The Honest Novelty Claim Now

S(M,T) is a constrained conditional logit, a form economics has had since 1974, instantiated for heterogeneous endpoint routing. What the paper contributes:

1. **The parameterization**: typed hard gates over heterogeneous endpoints (LLMs, agents, scripts, tools), 16 bounded compatibility features combined under a log-sum form, and a task-conditional cost temperature, all in one scoring interface.
2. **The analysis of that parameterization**: convergence under Hajek conditions for the online weight learner, O(sqrt(KN log N)) regret via contextual-bandit reduction, weight and temperature identifiability conditions, and the V2 refinements (headroom-capped composition, CVaR cost).
3. **The empirical program**: the measurement gap (0/14 attempts to ground hand-designed phi functions in 2.76M benchmark records), the learned bilinear resolution, the scaling ablation, and the 61-day production deployment with 94.7% measured cost reduction and 25.4% agent routing.

The equation is the vehicle, not the cargo. The cargo is the parameterization, the guarantees attached to it, and the measured behavior of a deployed instance.

---

## Sources

- [McFadden (1974), Conditional Logit Analysis of Qualitative Choice Behavior](https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf)
- [McFadden 1974 citation record](https://www.scirp.org/reference/referencespapers?referenceid=3157844)
- [Train (2009), Discrete Choice Methods with Simulation](https://eml.berkeley.edu/books/choice2.html)
- [Nested logit inclusive value formulation](https://gsbdbi.github.io/torch-choice/nested_logit_model_house_cooling/)
