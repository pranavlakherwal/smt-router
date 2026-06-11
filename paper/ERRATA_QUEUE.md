# S(M,T) Paper Errata Queue

Purpose: every adversarial finding against the published paper lands here and stays OPEN until the tex (and, where applicable, the published copies) are fixed. The Ramanujan nightly run re-finding the same issue for 2.5 months without a tex change is the failure mode this file exists to prevent.

## Schema

Each entry is a fenced block with these fields:

```
claim_id:      stable identifier (SMT-ERR-NNN)
attack:        what the adversarial finding asserts
evidence:      report citations, confidences, file:line where the paper makes the claim
claim_status:  upheld | weakened | refuted   (status of the PAPER's claim after the attack)
resolution:    what was or must be changed
status:        OPEN | RESOLVED
```

---

## Entries

```
claim_id:      SMT-ERR-001
attack:        Equation-form novelty. S(M,T) = [prod g_j] * [sum w_i phi_i] * exp(-beta.H) is,
               under log transform and softmax, a constrained conditional logit (McFadden 1974):
               gates = choice-set restriction (exact), cost term = linear-in-parameters utility
               (exact), compatibility term = one nonstandard log-sum regressor of nested-logit
               inclusive-value form (Train 2009). MoE gating (Jacobs 1991, Shazeer 2017) is ML
               prior art for the gate structure. The paper claimed the composition as the
               contribution without citing the 50-year-old form.
evidence:      ramanujan/2026-03-25.md item 12 (VERIFIED 98%); ramanujan/2026-05-23.md item 17
               (claim REFUTED 90%); ramanujan/2026-06-10.md items 5 (REFUTED 98%) and 12
               (VERIFIED 95%); 58 nightly reports total, first one day after the 2026-03-24
               publication. Paper claim sites: main.tex abstract; sections/01_intro.tex eq.
               paragraph; sections/08_related_work.tex "What S(M,T) Adds" item (1);
               sections/10_conclusion.tex opening.
claim_status:  refuted (as an equation-form novelty claim); the parameterization, analysis,
               and empirical contributions stand. Full algebra: MCFADDEN_RESOLUTION.md.
resolution:    RESOLVED-IN-TEX this session (2026-06-11): dedicated Discrete Choice Theory
               paragraph added to sections/08_related_work.tex citing mcfadden1974conditional,
               train2009discrete, jacobs1991adaptive, shazeer2017outrageously and stating the
               equivalence; novelty language repositioned in main.tex abstract,
               sections/01_intro.tex, sections/10_conclusion.tex, and related-work contribution
               item (1); bib entries verified present; PDF recompiled. Published Medium/Reddit/
               GitHub copies NOT yet updated (owner action).
status:        RESOLVED
```

```
claim_id:      SMT-ERR-002
attack:        Decision-count drift. The production decision count differs across sections:
               1,580 (main.tex abstract; sections/07d_production_validation.tex lines 4, 16,
               40-42, 84), 1,593 (07d "Three Views", lines 172-208, and
               sections/07e_endpoint_profiler.tex line 4), 1,622 (07e lines 62, 66), and 1,843
               (live shadow-router decision count per internal status docs, 2026-06-04 agent
               guild readiness workflow; not in the paper). The counts grew because later
               analyses ran on later database snapshots and earlier numbers were never
               backported or footnoted.
evidence:      grep of sections/*.tex this session confirms all three in-paper values and their
               exact locations; 1,843 appears only outside the paper.
claim_status:  weakened (numbers are internally inconsistent without a stated snapshot
               convention; no result depends on which snapshot is used, but a reader cannot
               tell that)
resolution:    To be reconciled in the arXiv v2 pass: pick one convention (either restate all
               analyses on a single frozen snapshot, or add a per-section snapshot-date
               footnote stating the count at each analysis date) and apply it across abstract,
               07d, and 07e.
status:        OPEN
```

```
claim_id:      SMT-ERR-003
attack:        Stale novelty-survival claim. sections/05_adversarial.tex line 86 states "All six
               devil's-advocate challenges were refuted at 85-95% confidence. The paper's
               novelty claims survive adversarial attack." Later Ramanujan runs contradict this
               for at least three of the six: the S(M,T) equation-form claim (REFUTED 90-98%,
               2026-05-23 and 2026-06-10), heterogeneous endpoint routing (REFUTED 95%,
               2026-06-10 items 13, 37), and bilinear phi novelty (REFUTED 95%, 2026-06-10
               items 28, 52).
evidence:      ramanujan/2026-06-10.md items 5, 13, 17, 28, 37, 52; ramanujan/2026-05-23.md
               item 17. Paper claim site: sections/05_adversarial.tex lines 85-86.
claim_status:  refuted (the sentence reports one early run as if it were the standing verdict)
resolution:    arXiv v2 pass: rewrite the "Bonus: Novelty Falsification" paragraph to report
               the verification honestly across runs, noting which challenges later succeeded
               and pointing to the Discrete Choice Theory paragraph (Section 8). Out of this
               session's file scope.
status:        OPEN
```
