//! Stub types for SGLang's Model Gateway. **REPLACE BEFORE UPSTREAM PR.**
//!
//! When this crate moves into `sgl-project/sglang` as
//! `sgl-model-gateway/src/policies/s_mt/`, these stubs vanish — the real
//! gateway crate exports `RoutingPolicy`, `WorkerHandle`, `RequestEnvelope`,
//! `RequestOutcome`, `WorkerPair`, and `RoutingError`.
//!
//! For now, they let `cargo check` and `cargo test` succeed in isolation so
//! the math, gate, learner, and state can be reviewed and tested standalone.

use async_trait::async_trait;

/// One serving worker in the gateway's pool.
#[derive(Debug, Clone)]
pub struct WorkerHandle {
    pub id_str: String,
    pub model_id: String,
    pub context_window: usize,
    pub safety_class: String,
    pub capabilities: Vec<String>,
    pub cost_input_per_m: f64,
    pub cost_output_per_m: f64,
    pub cost_cached_per_m: f64,
    pub supports_prefill_flag: bool,
    pub supports_decode_flag: bool,
}

impl WorkerHandle {
    pub fn id(&self) -> &str {
        &self.id_str
    }
    pub fn supports_prefill(&self) -> bool {
        self.supports_prefill_flag
    }
    pub fn supports_decode(&self) -> bool {
        self.supports_decode_flag
    }
}

#[derive(Debug, Clone)]
pub struct WorkerPair {
    pub prefill: WorkerHandle,
    pub decode: WorkerHandle,
}

/// An incoming inference request, as seen by the gateway.
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    pub request_id: String,
    pub prompt_token_count: usize,
    pub max_output_tokens: usize,
    pub model_hint: Option<String>,
    pub safety_class: String,
    pub task_hint: Option<String>,
    pub headers: std::collections::HashMap<String, String>,
}

/// An observed outcome from a completed request.
#[derive(Debug, Clone)]
pub struct RequestOutcome {
    pub request_id: String,
    pub latency_ms: f64,
    pub success: bool,
    pub quality_proxy: Option<f64>,  // optional downstream verifier or user signal
    pub cost_actual_usd: Option<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("no eligible worker matched the request")]
    NoEligibleWorker,
    #[error("policy state error: {0}")]
    State(String),
}

pub type RoutingResult<T> = Result<T, RoutingError>;

/// The trait every built-in policy implements. Stub mirrors the surface that
/// the SGLang research found in `sgl-model-gateway/src/policies/` (issue #7535).
#[async_trait]
pub trait RoutingPolicy: Send + Sync {
    fn name(&self) -> &'static str;
    async fn select_single(
        &self,
        req: &RequestEnvelope,
        workers: &[WorkerHandle],
    ) -> RoutingResult<WorkerHandle>;
    async fn select_pair(
        &self,
        req: &RequestEnvelope,
        workers: &[WorkerHandle],
    ) -> RoutingResult<WorkerPair>;
    /// **The seam.** No built-in policy uses this hook. We do.
    async fn on_request_complete(&self, outcome: &RequestOutcome) -> RoutingResult<()>;
}
