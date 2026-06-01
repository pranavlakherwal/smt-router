//! Cost estimation for the exp(−β · cost) term.

#[cfg(not(feature = "upstream"))]
use crate::sglang_stub::{RequestEnvelope, WorkerHandle};
#[cfg(feature = "upstream")]
use sglang_gateway::{RequestEnvelope, WorkerHandle};

use crate::{config::SmtConfig, features::PhiFeatures};

#[derive(Debug, Clone)]
pub struct RequestCostHint {
    pub estimated_dollars: f64,
}

#[derive(Debug, Clone)]
pub struct CostModel {
    default_output_tokens: usize,
    cache_hit_share: f64,
}

impl CostModel {
    pub fn from_config(cfg: &SmtConfig) -> Self {
        Self {
            default_output_tokens: cfg.default_output_tokens,
            cache_hit_share: cfg.assumed_cache_hit_share,
        }
    }

    /// Estimate dollar-cost of running `worker` on `req`. The worker exposes
    /// per-million-token pricing; we apply the cache-hit share as a discount
    /// proxy (in production this is replaced with a KV-residency estimate).
    pub fn estimate(
        &self,
        worker: &WorkerHandle,
        req: &RequestEnvelope,
        _phi: &PhiFeatures,
    ) -> f64 {
        let input = req.prompt_token_count as f64;
        let output = req.max_output_tokens.max(self.default_output_tokens) as f64;

        let cached_input = input * self.cache_hit_share;
        let fresh_input = input - cached_input;

        let c_in = fresh_input / 1_000_000.0 * worker.cost_input_per_m;
        let c_cached = cached_input / 1_000_000.0 * worker.cost_cached_per_m;
        let c_out = output / 1_000_000.0 * worker.cost_output_per_m;

        c_in + c_cached + c_out
    }
}
