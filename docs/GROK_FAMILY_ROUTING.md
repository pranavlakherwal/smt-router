# Worked Example: S(M,T) Across the Grok Model Family

> Grok is a publicly-priced model family with three independent routing dimensions: model tier, `reasoning_effort` axis, and a coding specialist (`grok-build-0.1`). This document walks S(M,T) across six representative query types and aggregates cost-vs-quality on a realistic traffic mix. All pricing is from docs.x.ai; all assumptions are explicit. The math is intentionally trivial enough for a reader to spot-check any cell.

## The candidate set

| Candidate (model + effort) | Context | $/M input | $/M cached | $/M output | Notes |
|---|---|---|---|---|---|
| grok-4.3 + {none,low,med,high} | 1M | 1.25 | 0.20 | 2.50 | frontier, multimodal |
| grok-4.20-reasoning + {low,med,high} | 1M | 1.25 | 0.20 | 2.50 | reasoning sibling |
| grok-4.20-non-reasoning | 1M | 1.25 | 0.20 | 2.50 | no-thinking sibling |
| grok-4.20-multi-agent + {4, 16} | 2M | 1.25 | 0.20 | 2.50 | parallel decomposition |
| grok-4.1-fast + {none,low,med,high} | 2M | 0.20 | 0.05 | 0.50 | cost leader |
| grok-build-0.1 | 256K | 1.00 | 0.20 | 2.00 | coding specialist |

`reasoning_effort` output-token verbosity multiplier (Artificial Analysis observed ~2.5× at "high"; we extrapolate):

| effort | output-token multiplier |
|---|---|
| none | 1.0× |
| low | 1.5× |
| med | 2.5× |
| high | 4.0× |

Quality prior (normalised to `grok-4.3-high = 1.00`):

| Candidate | Quality |
|---|---|
| grok-4.3-high | 1.00 |
| grok-4.3-med | 0.97 |
| grok-4.3-low | 0.93 |
| grok-4.3-none | 0.85 |
| grok-4.20-reasoning-high | 0.99 |
| grok-4.20-reasoning-med | 0.96 |
| grok-4.20-non-reasoning | 0.85 |
| grok-4.20-multi-agent (decomposable) | 0.98 |
| grok-4.1-fast-high | 0.88 |
| grok-4.1-fast-none | 0.78 |
| grok-build-0.1 (code) | 0.95 |
| grok-build-0.1 (non-code) | 0.70 |

We use `β = 0.4` (the default cost-decay coefficient). Weights match the personal-twin's converged Q2 2026 state, plus we let `phi_3_specialty` fire when a candidate declares a matching capability.

## Six query types, walked through

For each: tokens, the always-flagship-high cost, what S(M,T) picks and why, the picked cost, the cost saving, and the quality delta.

### 1. Simple factual — "What is the capital of France?"

- Tokens: 10 in, 10 out, no thinking needed.
- Eligible: every candidate (gate passes everywhere).
- Active features: `phi_1_relevance` low, `phi_8_composition` zero, `phi_3_specialty` zero.
- Cost-decay dominates because every other signal is weak. The cheapest tier wins.
- **Pick: `grok-4.1-fast-none`.**
- Always-flagship-high cost: 10/1M × 1.25 + (10 × 4)/1M × 2.50 = **$0.000113**
- Picked cost: 10/1M × 0.20 + 10/1M × 0.50 = **$0.0000070**
- **Saved: 93.8%.** Quality 0.78 nominal but task is trivial so effective parity. **Δq ≈ 0pp.**

### 2. Coding bug fix — "Fix the off-by-one in `parse_window()`"

- Tokens: 1500 in (code context) + 800 out.
- Gate passes all candidates; `grok-build-0.1` declares `code` capability so its `phi_3_specialty` fires.
- Active features: `phi_3_specialty` (code) high; `phi_1_relevance` moderate; `phi_8_composition` low.
- Two regimes by which weights are live:
  - **Weights as personal-twin converged (`phi_3 ≈ 0`):** `grok-4.1-fast-none` wins on pure cost-decay. Quality 0.88.
  - **Weights after specialty re-discovery (`phi_3 > 0`):** `grok-build-0.1` wins, specialty × moderate cost beats fast × low cost. Quality 0.95.
- Always-flagship-high: 1500/1M × 1.25 + (800 × 4)/1M × 2.50 = **$0.009875**
- `grok-4.1-fast-none`: 1500/1M × 0.20 + 800/1M × 0.50 = **$0.000700** → **93% saved, Δq -12pp**
- `grok-build-0.1`: 1500/1M × 1.00 + 800/1M × 2.00 = **$0.003100** → **69% saved, Δq -5pp**

### 3. Math reasoning — "Solve ∫ x²·eˣ dx"

- Tokens: 30 in, 600 out, needs reasoning.
- `phi_8_composition` fires (multi-step). Fast tier qualifies but quality cost is real on math.
- The reasoning sibling at med effort is a Pareto winner: same model class as flagship, lower verbosity.
- **Pick: `grok-4.20-reasoning-med`.**
- Always-flagship-high: 30/1M × 1.25 + (600 × 4)/1M × 2.50 = **$0.006038**
- Picked cost: 30/1M × 1.25 + (600 × 2.5)/1M × 2.50 = **$0.003788**
- **Saved: 37.3%, Δq -4pp.**

### 4. Multi-step refactor — "First rename, then extract a helper, then inline the old caller"

- Tokens: 100 in, 2000 out, structured plan.
- `phi_8_composition` very high (three steps). `phi_3_specialty` (code) also fires.
- Multi-agent variant looks tempting but is *more* expensive (16-agent variant multiplies output tokens by 16), so cost-decay penalises it.
- **Pick: `grok-build-0.1`** (specialty + low cost) over `grok-4.20-multi-agent` (composition + high cost).
- Always-flagship-high: 100/1M × 1.25 + (2000 × 4)/1M × 2.50 = **$0.020125**
- Picked cost: 100/1M × 1.00 + 2000/1M × 2.00 = **$0.004100**
- **Saved: 79.6%, Δq -5pp.**

### 5. Long-context RAG — "Summarise this 50K-token document"

- Tokens: 50,000 in, 500 out.
- Gate eliminates `grok-build-0.1` (256K context fits but the model declares code-only capability). All others fit.
- `phi_4_context_fit` fires; `phi_8_composition` low; summarisation is not reasoning-heavy.
- Cost-decay is brutal here: 50K input tokens make pricing the dominant signal.
- **Pick: `grok-4.1-fast-none`.**
- Always-flagship-high: 50000/1M × 1.25 + (500 × 4)/1M × 2.50 = **$0.06750**
- Picked cost: 50000/1M × 0.20 + 500/1M × 0.50 = **$0.010250**
- **Saved: 84.8%, Δq -3pp.** (Summarisation is forgiving; nominal 0.78 is effectively 0.90 on this task.)

### 6. Multimodal — "Analyse this satellite image, identify anomalies"

- Tokens: ~1200 in (image-tokens-equivalent + prompt), 500 out.
- Gate eliminates `grok-build-0.1` and likely `grok-4.20-multi-agent` (capability mismatch: no vision).
- `phi_7_multimodal` fires only on vision-capable candidates.
- High effort gives no quality lift for a one-shot vision read.
- **Pick: `grok-4.3-low`.**
- Always-flagship-high: 1200/1M × 1.25 + (500 × 4)/1M × 2.50 = **$0.006500**
- Picked cost: 1200/1M × 1.25 + (500 × 1.5)/1M × 2.50 = **$0.003375**
- **Saved: 48.1%, Δq -7pp.**

## Aggregate, representative traffic mix

Two scenarios depending on whether the learner re-discovers `phi_3_specialty` on the live distribution. On the personal-twin's traffic, `phi_3` zeroed out (the corpus is small and noisy on the specialty signal). On a heavier production traffic mix, the signal should reappear.

| Type | Share | Pick | Cost saved | Δq (pp) |
|---|---|---|---|---|
| Simple factual | 20% | grok-4.1-fast-none | 94% | 0 |
| Coding (specialty fires) | 35% | grok-build-0.1 | 69% | -5 |
| Coding (cost-only) | (alt) | grok-4.1-fast-none | 93% | -12 |
| Math / reasoning | 10% | grok-4.20-reasoning-med | 37% | -4 |
| Multi-step refactor | 5% | grok-build-0.1 | 80% | -5 |
| Long-context RAG | 10% | grok-4.1-fast-none | 85% | -3 |
| Multimodal | 5% | grok-4.3-low | 48% | -7 |
| Complex analysis (mixed) | 15% | grok-4.3-med | 25% | -3 |

**Weighted result.**

| Scenario | Cost saved | Quality delta |
|---|---|---|
| Specialty fires (`phi_3 > 0`), `grok-build-0.1` picked for code | **~65%** | **-3.5pp** |
| Cost-only (`phi_3 = 0`), fast picked for code | **~74%** | **-6pp** |

Both materially higher than the RouterBench v1.1 headline (25-30% cost cut at -2pp). The Grok lineup has a wider price spread (~100× combining the 25× model spread and the 4× reasoning-effort spread) than the RouterBench fleet, so the equation has more to work with.

## At fleet scale

Cost reduction scales linearly with inference spend at the per-query rate above. Indicative wedges for a single-org inference fleet operating at the ~65% point:

- $10M/month inference → ~$6.5M/month saved
- $100M/month inference → ~$65M/month saved
- $1B/month inference → ~$650M/month saved

These are not promises. They are the per-query rate applied at scale. Actual savings depend on traffic mix: more long-context and easy queries push savings up; more multimodal and complex-analysis traffic push them down.

## The quality side: same equation, cost-second

Set the gate to "highest-accuracy tier eligible" and let cost-decay break ties. The same math captures additional headroom toward the oracle ceiling (~96% on RouterBench v1.1) at higher cost. The operator picks the regime per SLA tier: cost-sensitive consumer traffic uses one β; quality-sensitive enterprise traffic uses another. One equation, two operating points.

## Why this works specifically on Grok

Three structural features of the lineup:

1. **The `reasoning_effort` axis is a free second dimension.** Same model with a 4× verbosity knob. The Pareto frontier in (cost, quality) space is dense.
2. **`grok-4.1-fast` exists at 16-25× cheaper than flagship** with quality competitive on simple and long-context tasks. Most traffic does not need the flagship.
3. **`grok-build-0.1` exists as a coding specialist** at 60-80% of flagship cost. Coding is a large traffic share for any agentic-tools surface.

A learned router exploits all three jointly. Per-knob hand-coding (4.3's `reasoning_effort`, 4.20's `agent_count`, Grok Build's subagent planner, gateway worker selection) exploits one axis at a time and leaves the cross-axis savings on the table.

## Repro

```bash
# The arithmetic is intentionally trivial so a reviewer can spot-check any cell.
# The pricing is live from docs.x.ai/developers/models.
# A runnable Python demo on a Grok-shaped candidate fleet lives in
# the canonical router/ package; see the repo README for entry points.
```

---

*Worked example for the S(M,T) framework. See `paper/main.tex` and `SMT_Router_Paper_v1.1.pdf` for the equation, the measurement-gap argument, and the RouterBench / RouteLLM results.*
