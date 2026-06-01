//! Integration tests for SmtPolicy. Uses the stub SGLang types.

use std::collections::HashMap;

use sgl_policy_smt::{
    SmtConfig, SmtPolicy,
    sglang_stub::{RequestEnvelope, RequestOutcome, RoutingPolicy as _, WorkerHandle},
};

fn make_worker(id: &str, ctx: usize, in_price: f64, out_price: f64, caps: &[&str]) -> WorkerHandle {
    WorkerHandle {
        id_str: id.to_string(),
        model_id: id.to_string(),
        context_window: ctx,
        safety_class: "general".to_string(),
        capabilities: caps.iter().map(|s| s.to_string()).collect(),
        cost_input_per_m: in_price,
        cost_output_per_m: out_price,
        cost_cached_per_m: in_price * 0.16,
        supports_prefill_flag: true,
        supports_decode_flag: true,
    }
}

#[tokio::test]
async fn cheap_worker_wins_on_short_prompt_when_caps_equal() {
    let workers = vec![
        make_worker("flagship", 1_000_000, 1.25, 2.50, &["code", "math", "general"]),
        make_worker("fast", 2_000_000, 0.20, 0.50, &["code", "general"]),
    ];
    let req = RequestEnvelope {
        request_id: "r1".into(),
        prompt_token_count: 200,
        max_output_tokens: 300,
        model_hint: None,
        safety_class: "general".into(),
        task_hint: Some("code".into()),
        headers: HashMap::new(),
    };
    let mut cfg = SmtConfig::default();
    cfg.initial_weights.insert("phi_1_relevance".into(), 1.0);
    cfg.beta = 1.0;
    let policy = SmtPolicy::new(cfg);
    let pick = policy.select_single(&req, &workers).await.unwrap();
    assert_eq!(pick.id(), "fast", "cost-decay should favor the cheaper worker");
}

#[tokio::test]
async fn gate_rejects_workers_with_too_small_context() {
    let workers = vec![
        make_worker("tiny", 1024, 0.10, 0.20, &["general"]),
        make_worker("big", 200_000, 3.00, 15.00, &["general"]),
    ];
    let req = RequestEnvelope {
        request_id: "r2".into(),
        prompt_token_count: 50_000,
        max_output_tokens: 500,
        model_hint: None,
        safety_class: "general".into(),
        task_hint: None,
        headers: HashMap::new(),
    };
    let mut cfg = SmtConfig::default();
    cfg.initial_weights.insert("phi_1_relevance".into(), 1.0);
    let policy = SmtPolicy::new(cfg);
    let pick = policy.select_single(&req, &workers).await.unwrap();
    assert_eq!(pick.id(), "big", "small context window should fail the gate");
}

#[tokio::test]
async fn on_request_complete_does_not_error_on_unknown_id() {
    let policy = SmtPolicy::new(SmtConfig::default());
    let outcome = RequestOutcome {
        request_id: "never-seen".into(),
        latency_ms: 300.0,
        success: true,
        quality_proxy: Some(0.9),
        cost_actual_usd: None,
    };
    let res = policy.on_request_complete(&outcome).await;
    assert!(res.is_ok(), "outcome for unknown request should be a no-op, not an error");
}
