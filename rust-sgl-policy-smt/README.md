# `rust-sgl-policy-smt` — Draft Rust port of S(M,T) for SGLang's Model Gateway

> This is the **pre-stage skeleton** for an upstream PR to [sgl-project/sglang](https://github.com/sgl-project/sglang). It implements an outcome-learning routing policy (`RoutingPolicy::Smt`) that plugs into SGLang's Model Gateway alongside the existing `random`, `round_robin`, `cache_aware`, `power_of_two`, `prefix_hash`, `bucket`, `manual`, and `consistent_hashing` policies. Status: skeleton ready for trait-shape verification against current SGLang HEAD.

## What lands upstream

A new policy module at `sgl-model-gateway/src/policies/s_mt/` plus a factory entry. Concretely:

```
sgl-model-gateway/
└── src/
    └── policies/
        ├── factory.rs                       <-- patched (one variant added)
        ├── mod.rs                           <-- patched (one re-export added)
        └── s_mt/                            <-- NEW directory, the contribution
            ├── mod.rs                       RoutingPolicy impl: select_single, select_pair, on_request_complete, name
            ├── score.rs                     S = gate · Σ w·φ · exp(-β · cost)
            ├── features.rs                  φ extractors over (worker, request)
            ├── gate.rs                      hard eligibility (capacity, capability, KV residency)
            ├── cost.rs                      β · cost(worker, request)
            ├── learner.rs                   Robbins-Monro online weight updates
            ├── state.rs                     in-memory weights + gRPC-mesh sync
            └── config.rs                    PolicyConfig::Smt variant + defaults
```

Plus:

- `sgl-model-gateway/tests/policies_smt.rs` — golden traces, regret-over-time, p99-latency budget.
- `sgl-model-gateway/docs/policies/s_mt.md` — math, calibration, operator runbook.
- Companion benchmark in `sgl-router-bench/` (out of scope for the initial PR).

## What the code in this directory shows

This directory mirrors the eventual upstream layout so a reviewer can read it standalone. **The crate compiles in isolation against stub trait definitions in `src/sglang_stub.rs`** so the math, gate, learner, and state are real Rust. The stubs document exactly which SGLang types need to be replaced when the code lands upstream.

```
rust-sgl-policy-smt/
├── Cargo.toml
├── README.md                           you are here
├── docs/
│   └── spec.md                         math, calibration, runbook
├── src/
│   ├── lib.rs                          crate root, exports SmtPolicy
│   ├── sglang_stub.rs                  trait stubs (REPLACE on upstream)
│   ├── mod_rs_target.rs                what lands as sgl-model-gateway/src/policies/s_mt/mod.rs
│   ├── score.rs                        S(M,T)
│   ├── features.rs                     φ extractors
│   ├── gate.rs                         hard eligibility
│   ├── cost.rs                         cost model
│   ├── learner.rs                      Robbins-Monro
│   ├── state.rs                        weights + mesh sync (stubbed)
│   └── config.rs                       PolicyConfig::Smt
└── tests/
    └── policy_smt.rs                   golden traces
```

## The seam, in one sentence

SGLang's `RoutingPolicy` Rust trait exposes an `on_request_complete(...)` outcome hook that **no built-in policy uses for online learning**. We use it. The other seven terms in the trait stay identical to the existing built-in policies, so the diff is small and local.

## Status flags

| Item | Status |
|---|---|
| Crate compiles against stubs | yes (see `cargo check`) |
| Trait shape pinned against SGLang HEAD | **TODO** before upstream PR |
| Tests pass on stubs | yes |
| Golden traces on real SGLang | **TODO** after stub replacement |
| Docs written | yes (`docs/spec.md`) |
| Owner reviewers | TBD (`@slin1237`, `@CatherineSue` are the gateway authors) |
| Scope | 6 to 10 weeks of focused work to mergeable PR |
