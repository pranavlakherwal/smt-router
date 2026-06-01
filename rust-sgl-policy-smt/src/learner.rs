//! Robbins-Monro stochastic-approximation online learner.
//!
//! Step size a_k = max(1/(k+1), floor). Satisfies the Robbins-Monro conditions:
//!
//! ```text
//!     Σ a_k = ∞    (slow enough to find the minimum)
//!     Σ a_k² < ∞   (fast enough to converge)
//! ```

#[cfg(not(feature = "upstream"))]
use crate::sglang_stub::RequestOutcome;
#[cfg(feature = "upstream")]
use sglang_gateway::RequestOutcome;

use std::collections::HashMap;

use crate::features::PhiFeatures;

#[derive(Debug, Clone)]
pub struct OutcomeSignal {
    pub latency_ms: f64,
    pub quality_proxy: f64, // [0,1]
    pub success: bool,
}

impl OutcomeSignal {
    pub fn from_envelope(outcome: &RequestOutcome) -> Self {
        Self {
            latency_ms: outcome.latency_ms,
            quality_proxy: outcome.quality_proxy.unwrap_or(0.5),
            success: outcome.success,
        }
    }

    /// Compose a scalar reward in [0, 1].
    /// 50% quality, 25% latency-SLA, 25% success.
    pub fn reward(&self, latency_sla_ms: f64) -> f64 {
        if !self.success {
            return 0.0;
        }
        let latency_score = if self.latency_ms <= latency_sla_ms {
            1.0
        } else {
            (1.0 - (self.latency_ms - latency_sla_ms) / latency_sla_ms).max(0.0)
        };
        0.5 * self.quality_proxy + 0.25 * latency_score + 0.25
    }
}

#[derive(Debug, Clone)]
pub struct RobbinsMonroLearner {
    weights: HashMap<String, f64>,
    k: usize,
    step_size_floor: f64,
    latency_sla_ms: f64,
}

impl RobbinsMonroLearner {
    pub fn new(initial_weights: HashMap<String, f64>) -> Self {
        Self {
            weights: initial_weights,
            k: 0,
            step_size_floor: 1e-4,
            latency_sla_ms: 5000.0,
        }
    }

    pub fn weights(&self) -> &HashMap<String, f64> {
        &self.weights
    }

    /// Project all weights to ≥ 0 (S(M,T) is positivist).
    fn clamp_nonneg(&mut self) {
        for v in self.weights.values_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    /// One online step. φ is the feature vector for the decision that
    /// produced this outcome; `signal` is the observed outcome.
    pub fn update(&mut self, phi: &PhiFeatures, signal: &OutcomeSignal) {
        let reward = signal.reward(self.latency_sla_ms);
        // ŝ = clamp(weighted_sum) under current weights.
        let mut s_hat: f64 = phi
            .iter()
            .map(|(name, value)| self.weights.get(name).copied().unwrap_or(0.0) * value)
            .sum();
        s_hat = s_hat.clamp(0.0, 1.0);

        self.k += 1;
        let a_k = (1.0 / (self.k as f64 + 1.0)).max(self.step_size_floor);
        let error = reward - s_hat;

        for (name, value) in phi.iter() {
            let entry = self.weights.entry(name.to_string()).or_insert(0.0);
            *entry += a_k * error * value;
        }
        self.clamp_nonneg();
    }

    pub fn decision_count(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_zero_on_failure() {
        let s = OutcomeSignal { latency_ms: 100.0, quality_proxy: 1.0, success: false };
        assert_eq!(s.reward(5000.0), 0.0);
    }

    #[test]
    fn update_moves_weight_toward_reward() {
        let mut learner = RobbinsMonroLearner::new(HashMap::new());
        let phi = PhiFeatures {
            phi_1_relevance: 1.0,
            ..PhiFeatures::default()
        };
        let signal = OutcomeSignal { latency_ms: 500.0, quality_proxy: 1.0, success: true };
        learner.update(&phi, &signal);
        let w1 = *learner.weights().get("phi_1_relevance").unwrap();
        assert!(w1 > 0.0);
    }
}
