//! Hard eligibility gate. Returns `passes: bool` per (worker, request).
//!
//! The gate carries most of the production lift. It eliminates 60-80% of
//! candidates before scoring; the weighted-feature score discriminates
//! among the survivors.

#[cfg(not(feature = "upstream"))]
use crate::sglang_stub::{RequestEnvelope, WorkerHandle};
#[cfg(feature = "upstream")]
use sglang_gateway::{RequestEnvelope, WorkerHandle};

use crate::{config::SmtConfig, features::PhiFeatures};

#[derive(Debug, Clone)]
pub struct GateResult {
    pub passes: bool,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub struct Gate {
    output_reserve_tokens: usize,
}

impl Gate {
    pub fn from_config(cfg: &SmtConfig) -> Self {
        Self {
            output_reserve_tokens: cfg.output_reserve_tokens,
        }
    }

    pub fn check(
        &self,
        worker: &WorkerHandle,
        req: &RequestEnvelope,
        _phi: &PhiFeatures,
    ) -> GateResult {
        // 1. Context window headroom.
        let needed = req.prompt_token_count + self.output_reserve_tokens.max(req.max_output_tokens);
        if needed > worker.context_window {
            return GateResult { passes: false, reason: "context-overflow" };
        }

        // 2. Safety class compatibility.
        if req.safety_class != "general" && req.safety_class != worker.safety_class {
            return GateResult { passes: false, reason: "safety-mismatch" };
        }

        // 3. Capability flag (if the request hints a task type and the worker
        //    declares capabilities, require a match).
        if let Some(task) = req.task_hint.as_deref() {
            if !worker.capabilities.is_empty()
                && !worker.capabilities.iter().any(|c| c == task)
            {
                return GateResult { passes: false, reason: "capability-missing" };
            }
        }

        GateResult { passes: true, reason: "ok" }
    }
}
