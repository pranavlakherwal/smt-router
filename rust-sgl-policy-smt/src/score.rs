//! The S(M,T) scoring function in one place.
//!
//! ```text
//!     S(M,T) = gate · Σᵢ wᵢ · φᵢ(M,T) · exp(−β · cost(M,T))
//! ```
//!
//! `gate` is applied upstream (caller returns 0 early if the gate fails). This
//! module computes the second and third terms together.

/// Compose the weighted-feature sum and cost-decay into one scalar score.
///
/// `weighted_sum` is Σᵢ wᵢ · φᵢ(M,T), already collapsed by the caller.
/// `cost` is the candidate's economic penalty (dollar-equivalents).
/// `beta` controls how aggressively cost discounts score (higher = more cost-averse).
#[inline]
pub fn compute(weighted_sum: f64, cost: f64, beta: f64) -> f64 {
    weighted_sum * (-beta * cost).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_decays_with_cost() {
        let cheap = compute(1.0, 0.0, 1.0);
        let pricey = compute(1.0, 2.0, 1.0);
        assert!(cheap > pricey);
        assert!((pricey / cheap - (-2.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn score_zero_when_weighted_sum_zero() {
        assert_eq!(compute(0.0, 0.5, 0.4), 0.0);
    }

    #[test]
    fn higher_beta_is_more_cost_averse() {
        let s_low_beta = compute(1.0, 1.0, 0.1);
        let s_high_beta = compute(1.0, 1.0, 1.0);
        assert!(s_low_beta > s_high_beta);
    }
}
