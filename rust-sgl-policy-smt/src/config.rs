//! `PolicyConfig::Smt` variant. Lands in the gateway's PolicyConfig enum
//! and is what `factory.rs` matches on.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Operator-tunable parameters for the S(M,T) policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtConfig {
    /// Cost-decay coefficient. Higher = more cost-averse.
    #[serde(default = "default_beta")]
    pub beta: f64,

    /// Reserved output tokens used by the gate to estimate context headroom.
    #[serde(default = "default_output_reserve")]
    pub output_reserve_tokens: usize,

    /// Default expected output tokens when the request does not declare one.
    #[serde(default = "default_output_tokens")]
    pub default_output_tokens: usize,

    /// Assumed prefix-cache hit share for cost estimation. In production this
    /// is replaced by a per-request KV-residency estimate from the gateway.
    #[serde(default = "default_cache_hit_share")]
    pub assumed_cache_hit_share: f64,

    /// Seed weights. May be empty; the learner will discover them online.
    #[serde(default)]
    pub initial_weights: HashMap<String, f64>,

    /// Latency SLA used to score the latency component of reward.
    #[serde(default = "default_latency_sla_ms")]
    pub latency_sla_ms: f64,
}

impl Default for SmtConfig {
    fn default() -> Self {
        Self {
            beta: default_beta(),
            output_reserve_tokens: default_output_reserve(),
            default_output_tokens: default_output_tokens(),
            assumed_cache_hit_share: default_cache_hit_share(),
            initial_weights: HashMap::new(),
            latency_sla_ms: default_latency_sla_ms(),
        }
    }
}

fn default_beta() -> f64 { 0.4 }
fn default_output_reserve() -> usize { 1024 }
fn default_output_tokens() -> usize { 500 }
fn default_cache_hit_share() -> f64 { 0.05 }
fn default_latency_sla_ms() -> f64 { 5000.0 }
