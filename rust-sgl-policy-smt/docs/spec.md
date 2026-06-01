# `RoutingPolicy::Smt` — Specification

## Math

```
S(M,T) = gate(M,T) · Σᵢ wᵢ · φᵢ(M,T) · exp(−β · cost(M,T))
```

| Term | Type | Role |
|---|---|---|
| `gate(M,T)` | `{0, 1}` | Hard eligibility: context, safety, capability |
| `Σᵢ wᵢ · φᵢ(M,T)` | `f64 ≥ 0` | Learned signal under online updates |
| `exp(−β · cost(M,T))` | `(0, 1]` | Economic penalty; β is operator-tunable |

The policy returns the argmax over eligible workers. Ties break on cost (cheaper wins).

## The seam

SGLang's `RoutingPolicy` trait exposes:

```rust
async fn on_request_complete(&self, outcome: &RequestOutcome) -> RoutingResult<()>;
```

No built-in policy uses this hook. `SmtPolicy` uses it to update weights via
Robbins-Monro stochastic approximation:

```text
w_i ← w_i + a_k · (reward − ŝ) · φ_i
a_k = max(1/(k+1), floor)
ŝ   = clamp(Σ_i w_i · φ_i, 0, 1)
```

`reward ∈ [0, 1]` is composed as `0.5·quality + 0.25·latency_score + 0.25·success`.

Step sizes satisfy the Robbins-Monro conditions (`Σ a_k = ∞`, `Σ a_k² < ∞`), so
the policy converges to a fixed point under stationary traffic, and tracks
non-stationary traffic with bounded lag set by the step-size floor.

## Calibration

Defaults shipped:

| Parameter | Default | Notes |
|---|---|---|
| `beta` | `0.4` | Operator-tunable; raise for cost-pressured tiers (consumer) |
| `output_reserve_tokens` | `1024` | Gate headroom for response tokens |
| `default_output_tokens` | `500` | Cost-estimate fallback when not declared |
| `assumed_cache_hit_share` | `0.05` | Replaced by real KV-residency estimate in production |
| `latency_sla_ms` | `5000` | Used by reward composition |
| `initial_weights` | `{}` | Empty by default; can be seeded from a prior policy |

The empirical result on RouterBench (n=73,008) is that two features dominate
under online learning on diverse traffic: `phi_1_relevance` (~0.47) and
`phi_8_composition` (~0.53). All others zero out. The operator can pre-seed
these for a warm start.

## Operator runbook

### Enable the policy

```toml
# gateway.toml
[policy]
name = "smt"

[policy.smt]
beta = 0.4
output_reserve_tokens = 1024
default_output_tokens = 500
latency_sla_ms = 5000

[policy.smt.initial_weights]
phi_1_relevance = 0.47
phi_8_composition = 0.53
```

### Observe weights drift

The policy exposes its weights via the standard gateway admin endpoint
(`/admin/policy/state`). Recommended Prometheus metrics to scrape:

- `sgl_policy_smt_weights{feature}` — current weight per feature
- `sgl_policy_smt_decision_count` — total observed outcomes
- `sgl_policy_smt_reward_ema` — exponential moving average of observed reward

### When to tune `beta`

- Inference economics under pressure: raise `beta` to push more traffic to cheaper workers. Watch reward EMA: if it drops more than ~5%, you have gone too far.
- Quality-sensitive tier (enterprise, IL5): lower `beta` toward 0.1 and rely on the gate for cost ceilings instead.

### Failure modes and mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| All weights zero out | Same worker picked every time after enough outcomes | Lower `step_size_floor`; check that `quality_proxy` is non-trivial |
| Policy chases noise | High variance in worker picks over time | Raise `step_size_floor` to slow the learner |
| Always picks flagship | `phi_1_relevance` saturated | Verify the gate is doing real work; check `gates_failed` rate |
| Cold start unstable | First N outcomes give wild reward | Seed `initial_weights` from a prior policy |

## What lands in the upstream PR

```
sgl-model-gateway/src/policies/factory.rs        + 1 match arm for PolicyConfig::Smt
sgl-model-gateway/src/policies/mod.rs            + 1 line: pub mod s_mt;
sgl-model-gateway/src/policies/s_mt/             NEW directory (this crate's src/)
sgl-model-gateway/tests/policies_smt.rs          NEW: golden traces
sgl-model-gateway/docs/policies/s_mt.md          NEW: this spec
sgl-model-gateway/CHANGELOG.md                   + line: "feat(policies): add s_mt"
```

Estimated review surface: ~800 lines net new, plus ~5 lines patched.
