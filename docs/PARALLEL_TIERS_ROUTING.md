# Worked example: S(M,T) over a tiered Task API

This note works the S(M,T) routing score against a **tiered, fixed-price task
API** instead of a fleet of token-priced chat models. The public price card of
Parallel.ai's Task API is used as a concrete, real instance. Nothing here is a
measured benchmark of that service: the per-class quality priors are
*illustrative* numbers chosen to show the mechanism. The arithmetic is
reproducible from `examples/parallel_tier_routing.py`.

## 1. Why a tiered API is a clean case for S(M,T)

The core scoring equation is unchanged:

```
S(tier, T) = gate(tier, T) · Σ wᵢ·φᵢ(tier, T) · exp(−β · cost(tier))
```

A tiered API collapses the feature sum to a single dominant feature, the
per-class quality prior `q(tier, T)`, so the score reduces to:

```
S(tier, T) = q(tier, T) · exp(−β · cost(tier))
```

This is the smallest non-trivial instance of the equation. There is one knob
that matters (β), and the candidate set is a totally ordered price ladder, so
every routing decision is auditable by hand. That makes it the right place to
verify the math before adding feature richness back in.

## 2. The price ladder

Parallel.ai Task API list price, converted to dollars per single request
(`price_per_1k / 1000`):

| Tier | $/1K req | $/req |
|---|---|---|
| Lite | 5 | 0.005 |
| Base | 10 | 0.010 |
| Core | 25 | 0.025 |
| Core2x | 50 | 0.050 |
| Pro | 100 | 0.100 |
| Ultra | 300 | 0.300 |
| Ultra2x | 600 | 0.600 |
| Ultra4x | 1200 | 1.200 |
| Ultra8x | 2400 | 2.400 |

Top-to-bottom spread is 480×, but the absolute numbers are tiny. That single
fact sets the useful range of β (see §6).

## 3. Quality priors per difficulty class (illustrative)

`q(tier, T)` is the probability the tier returns an acceptable answer for a
query of that difficulty class. Easy queries saturate early; hard queries need
depth.

| Tier | Easy | Medium | Hard |
|---|---|---|---|
| Lite | 0.90 | 0.45 | 0.20 |
| Base | 0.95 | 0.62 | 0.30 |
| Core | 0.97 | 0.80 | 0.55 |
| Core2x | 0.98 | 0.85 | 0.65 |
| Pro | 0.99 | 0.92 | 0.80 |
| Ultra | 1.00 | 0.97 | 0.92 |
| Ultra2x | 1.00 | 0.98 | 0.96 |
| Ultra4x | 1.00 | 0.985 | 0.98 |
| Ultra8x | 1.00 | 0.99 | 0.99 |

- Easy: "what city is company X headquartered in"
- Medium: "find 5 startups in a niche, with funding and team size"
- Hard: "full competitive teardown of a market, with pricing and defensibility"

## 4. Winner per class at four β operating points

| Class | β=0.4 | β=4 | β=8 | β=20 |
|---|---|---|---|---|
| Easy | Core2x | Base | Base | Lite |
| Medium | Pro | Core | Core | Base |
| Hard | Ultra | Pro | Core | Core |

Two sanity reads: at small β the router buys near the top of the ladder
(quality dominates); as β grows it walks down the ladder class by class, and the
harder the class the later it gives ground.

## 5. Four properties that must hold

These are checked programmatically in `examples/parallel_tier_routing.py`; all
four pass.

- **P1 (quality limit).** As β → 0 the cost term → 1, so `argmax S` is the
  highest-quality tier. Verified: every class selects its max-quality tier.
- **P2 (cost limit).** As β → ∞ the score collapses to 0 fastest for the
  dearest tiers, so `argmax S` is the cheapest tier. Verified: every class
  selects Lite.
- **P3 (cost monotonicity).** For two tiers of equal quality, the cheaper one
  scores at least as high at every β > 0. Verified on the Ultra vs Ultra8x pair
  at q = 1.0.
- **P4 (difficulty monotonicity).** At a fixed β, a harder class selects a tier
  at least as expensive as an easier class. Verified at β = 8:
  easy $0.010 ≤ medium $0.025 ≤ hard $0.025.

## 6. The β calibration insight

Because per-request prices are tiny, `exp(−β·cost)` only bends where
`β·cost ≈ O(1)`. With cost in the $0.005–$0.30 band, that puts the *useful* β
range far below what a per-token chat router would use. The mixed-workload
frontier (60% easy / 30% medium / 10% hard) makes the trade explicit:

| β | routed $/req | save vs always-Ultra | quality | Δ quality vs Ultra |
|---|---|---|---|---|
| 0.2 | 0.1800 | 40.0% | 0.977 | −0.6 pp |
| 0.4 | 0.0900 | 70.0% | 0.956 | −2.7 pp |
| 1.0 | 0.0550 | 81.7% | 0.938 | −4.5 pp |
| 2.0 | 0.0310 | 89.7% | 0.905 | −7.8 pp |
| 4.0 | 0.0235 | 92.2% | 0.890 | −9.3 pp |
| 8.0 | 0.0160 | 94.7% | 0.865 | −11.8 pp |
| 20 | 0.0085 | 97.2% | 0.781 | −20.2 pp |

Always-Ultra baseline on this mix: $0.3000/req at quality 0.983.

The knee for a near-baseline SLA sits at **β ≈ 0.2–0.4**: roughly **40–70% lower
cost while holding within ~2pp of always-Ultra quality**, because most real
queries do not need the top tier. This is the mechanism, on illustrative priors.
It is not a measured result.

## 7. Why β is learned, not set

The frontier above is a static sweep. In a live system you never hand-pick β.
The operator sets a quality floor (say, "stay within 2pp of always-top"), and
the weights and β move online from realised outcomes through the
`on_request_complete` hook, the same Robbins-Monro update used by the production
Shadow Router. The router converges to the point on the frontier that holds the
floor at least cost, and it tracks drift as query mix and tier quality change
over time.

## 8. Reproduce

```
python3 examples/parallel_tier_routing.py
```

Prints the full score tables, the four property checks, and the frontier.
