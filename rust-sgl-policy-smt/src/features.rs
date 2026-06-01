//! Feature extraction for φ_i(M,T).
//!
//! Sixteen candidate features in the spec; under online learning on production
//! traffic, weights converged on two: relevance (φ_1) and composition headroom (φ_8).
//! All sixteen are still extracted so the learner can re-discover signal on new
//! distributions.

#[cfg(not(feature = "upstream"))]
use crate::sglang_stub::RequestEnvelope;
#[cfg(feature = "upstream")]
use sglang_gateway::RequestEnvelope;

/// One feature vector for a (request, candidate) pair. Sparse by design.
#[derive(Debug, Clone, Default)]
pub struct PhiFeatures {
    pub phi_1_relevance: f64,
    pub phi_2_recency: f64,
    pub phi_3_specialty: f64,
    pub phi_4_context_fit: f64,
    pub phi_5_latency_prior: f64,
    pub phi_6_quality_prior: f64,
    pub phi_7_multimodal: f64,
    pub phi_8_composition: f64,
    pub phi_9_cost_aware: f64,
    pub phi_10_safety: f64,
    pub phi_11_reliability: f64,
    pub phi_12_freshness: f64,
    pub phi_13_complexity: f64,
    pub phi_14_session: f64,
    pub phi_15_user_pref: f64,
    pub phi_16_provider: f64,
    pub task_hint: Option<String>,
}

impl PhiFeatures {
    /// Iterate (name, value) over the numeric features.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, f64)> {
        let v = [
            ("phi_1_relevance", self.phi_1_relevance),
            ("phi_2_recency", self.phi_2_recency),
            ("phi_3_specialty", self.phi_3_specialty),
            ("phi_4_context_fit", self.phi_4_context_fit),
            ("phi_5_latency_prior", self.phi_5_latency_prior),
            ("phi_6_quality_prior", self.phi_6_quality_prior),
            ("phi_7_multimodal", self.phi_7_multimodal),
            ("phi_8_composition", self.phi_8_composition),
            ("phi_9_cost_aware", self.phi_9_cost_aware),
            ("phi_10_safety", self.phi_10_safety),
            ("phi_11_reliability", self.phi_11_reliability),
            ("phi_12_freshness", self.phi_12_freshness),
            ("phi_13_complexity", self.phi_13_complexity),
            ("phi_14_session", self.phi_14_session),
            ("phi_15_user_pref", self.phi_15_user_pref),
            ("phi_16_provider", self.phi_16_provider),
        ];
        v.into_iter()
    }
}

#[derive(Debug, Default)]
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract φ from a request envelope. In the upstream PR this also takes
    /// a worker handle so per-(worker, request) features (latency_prior,
    /// quality_prior, prefix-hit estimate) are computed; here we keep it
    /// request-only for the skeleton.
    pub fn extract(&self, req: &RequestEnvelope) -> PhiFeatures {
        let mut phi = PhiFeatures::default();
        let n = req.prompt_token_count as f64;

        // φ_1 relevance: length-normalized proxy.
        phi.phi_1_relevance = (n / 1000.0).min(1.0);

        // φ_4 context fit: decays as prompt grows toward 100k tokens.
        phi.phi_4_context_fit = (1.0 - (n / 100_000.0)).max(0.0);

        // φ_8 composition headroom: keyed off the task_hint header if present.
        // In production: derived from prompt parsing for "then", "after", "finally".
        if let Some(t) = req.task_hint.as_deref() {
            phi.phi_8_composition = match t {
                "multi-step" | "agent" | "plan" => 0.9,
                "single" | "completion" => 0.1,
                _ => 0.3,
            };
        }

        // φ_13 complexity: prompt-length proxy in [0,1].
        phi.phi_13_complexity = (n / 5000.0).min(1.0);

        phi.task_hint = req.task_hint.clone();
        phi
    }
}
