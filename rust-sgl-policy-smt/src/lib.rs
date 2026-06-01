//! `sgl-policy-smt` — S(M,T) routing policy for SGLang's Model Gateway.
//!
//! The math:
//!
//! ```text
//!     S(M,T) = gate(M,T) · Σᵢ wᵢ · φᵢ(M,T) · exp(−β · cost(M,T))
//! ```
//!
//! Three terms:
//!
//! 1. `gate(M,T)`: hard eligibility (0 or 1). Capacity, capability, KV residency.
//! 2. `Σᵢ wᵢ · φᵢ(M,T)`: weighted features, learned online via Robbins-Monro.
//! 3. `exp(−β · cost(M,T))`: cost-decay penalty.
//!
//! ## Where this lands
//!
//! When upstreamed, this crate's modules move into
//! `sgl-model-gateway/src/policies/s_mt/`, the `SmtPolicy` impl wires into
//! the gateway's `RoutingPolicy` trait, and a single line is added to
//! `sgl-model-gateway/src/policies/factory.rs` to register it.
//!
//! ## The seam
//!
//! `RoutingPolicy::on_request_complete` is the outcome hook nothing else uses.
//! We use it to update weights via [`RobbinsMonroLearner::update`] on every
//! observed reward.

pub mod config;
pub mod cost;
pub mod features;
pub mod gate;
pub mod learner;
pub mod score;
pub mod state;

#[cfg(not(feature = "upstream"))]
pub mod sglang_stub;
#[cfg(not(feature = "upstream"))]
use sglang_stub as sglang;

#[cfg(feature = "upstream")]
use sglang_gateway as sglang;

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use tracing::{debug, instrument};

pub use config::SmtConfig;
pub use cost::{CostModel, RequestCostHint};
pub use features::{FeatureExtractor, PhiFeatures};
pub use gate::{Gate, GateResult};
pub use learner::{OutcomeSignal, RobbinsMonroLearner};
pub use state::{PolicyState, WeightVector};

use sglang::{
    RequestEnvelope, RequestOutcome, RoutingPolicy, RoutingResult, WorkerHandle, WorkerPair,
};

/// The S(M,T) policy. Implements SGLang's `RoutingPolicy`.
pub struct SmtPolicy {
    config: SmtConfig,
    state: Arc<RwLock<PolicyState>>,
    learner: Arc<RwLock<RobbinsMonroLearner>>,
    features: FeatureExtractor,
    cost_model: CostModel,
    gate: Gate,
}

impl SmtPolicy {
    pub fn new(config: SmtConfig) -> Self {
        let initial_weights = config.initial_weights.clone();
        Self {
            features: FeatureExtractor::default(),
            cost_model: CostModel::from_config(&config),
            gate: Gate::from_config(&config),
            state: Arc::new(RwLock::new(PolicyState::with_weights(initial_weights.clone()))),
            learner: Arc::new(RwLock::new(RobbinsMonroLearner::new(initial_weights))),
            config,
        }
    }

    fn score_worker(&self, worker: &WorkerHandle, req: &RequestEnvelope, phi: &PhiFeatures) -> f64 {
        let gate_result = self.gate.check(worker, req, phi);
        if !gate_result.passes {
            return 0.0;
        }
        let weights = self.state.read().weights().snapshot();
        let weighted_sum: f64 = phi
            .iter()
            .map(|(name, value)| weights.get(name).copied().unwrap_or(0.0) * value)
            .sum();
        let cost = self.cost_model.estimate(worker, req, phi);
        score::compute(weighted_sum, cost, self.config.beta)
    }
}

#[async_trait]
impl RoutingPolicy for SmtPolicy {
    fn name(&self) -> &'static str {
        "smt"
    }

    /// Pick a single worker for this request.
    #[instrument(skip(self, req, workers), fields(policy = "smt"))]
    async fn select_single(
        &self,
        req: &RequestEnvelope,
        workers: &[WorkerHandle],
    ) -> RoutingResult<WorkerHandle> {
        let phi = self.features.extract(req);
        let mut scored: Vec<(WorkerHandle, f64)> = workers
            .iter()
            .map(|w| (w.clone(), self.score_worker(w, req, &phi)))
            .collect();

        // Sort by score desc; tiebreak by per-worker cost asc.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ca = self.cost_model.estimate(&a.0, req, &phi);
                    let cb = self.cost_model.estimate(&b.0, req, &phi);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        debug!(top_count = scored.len(), top_score = scored.first().map(|x| x.1));

        scored
            .into_iter()
            .next()
            .map(|(w, _s)| w)
            .ok_or_else(|| sglang::RoutingError::NoEligibleWorker)
    }

    /// Pick a prefill/decode pair for PD disaggregation.
    async fn select_pair(
        &self,
        req: &RequestEnvelope,
        workers: &[WorkerHandle],
    ) -> RoutingResult<WorkerPair> {
        // Score every worker once.
        let phi = self.features.extract(req);
        let mut scored: Vec<(WorkerHandle, f64)> = workers
            .iter()
            .map(|w| (w.clone(), self.score_worker(w, req, &phi)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Prefill: highest score that is prefill-capable.
        // Decode: highest remaining score that is decode-capable.
        let prefill = scored
            .iter()
            .find(|(w, _)| w.supports_prefill())
            .cloned()
            .ok_or(sglang::RoutingError::NoEligibleWorker)?
            .0;
        let decode = scored
            .iter()
            .find(|(w, _)| w.supports_decode() && w.id() != prefill.id())
            .cloned()
            .ok_or(sglang::RoutingError::NoEligibleWorker)?
            .0;

        Ok(WorkerPair { prefill, decode })
    }

    /// THE SEAM. Outcome ingest. Updates weights via Robbins-Monro.
    ///
    /// No other built-in SGLang policy uses this hook. We use it to close
    /// the learning loop: every observed outcome shifts the policy.
    async fn on_request_complete(&self, outcome: &RequestOutcome) -> RoutingResult<()> {
        let phi = match self.state.read().recall_features(&outcome.request_id) {
            Some(phi) => phi,
            None => return Ok(()), // outcome for an unknown decision; drop.
        };
        let signal = OutcomeSignal::from_envelope(outcome);
        self.learner.write().update(&phi, &signal);
        // Mirror the new weights into shared policy state for `score_worker` to pick up.
        self.state.write().set_weights(self.learner.read().weights().clone());
        debug!(request_id = %outcome.request_id, "weights updated");
        Ok(())
    }
}
