//! Shared policy state: current weights, per-request feature memo.
//!
//! For the upstream PR, `mesh_sync` becomes a real gRPC stream against the
//! Gateway 3.0 plan in issue #13098 (cross-replica weight sync). Here it's
//! a stub that returns OK so the policy can be reviewed and tested.

use std::collections::HashMap;

use crate::features::PhiFeatures;

#[derive(Debug, Clone, Default)]
pub struct WeightVector(HashMap<String, f64>);

impl WeightVector {
    pub fn new(weights: HashMap<String, f64>) -> Self {
        Self(weights)
    }
    pub fn get(&self, name: &str) -> Option<&f64> {
        self.0.get(name)
    }
    pub fn snapshot(&self) -> HashMap<String, f64> {
        self.0.clone()
    }
}

#[derive(Debug)]
pub struct PolicyState {
    weights: WeightVector,
    /// Per-request feature memo so on_request_complete can recover the φ
    /// that produced the original decision.
    feature_memo: HashMap<String, PhiFeatures>,
    /// How many feature memos to retain (LRU-ish; in production this is bounded).
    memo_capacity: usize,
    memo_order: Vec<String>,
}

impl PolicyState {
    pub fn with_weights(weights: HashMap<String, f64>) -> Self {
        Self {
            weights: WeightVector::new(weights),
            feature_memo: HashMap::new(),
            memo_capacity: 10_000,
            memo_order: Vec::new(),
        }
    }

    pub fn weights(&self) -> &WeightVector {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: HashMap<String, f64>) {
        self.weights = WeightVector::new(weights);
    }

    pub fn remember_features(&mut self, request_id: String, phi: PhiFeatures) {
        if self.feature_memo.len() >= self.memo_capacity {
            // Evict oldest.
            if let Some(oldest) = self.memo_order.first().cloned() {
                self.feature_memo.remove(&oldest);
                self.memo_order.remove(0);
            }
        }
        self.feature_memo.insert(request_id.clone(), phi);
        self.memo_order.push(request_id);
    }

    pub fn recall_features(&self, request_id: &str) -> Option<PhiFeatures> {
        self.feature_memo.get(request_id).cloned()
    }
}
