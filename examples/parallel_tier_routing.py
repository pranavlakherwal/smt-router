#!/usr/bin/env python3
"""
Logical verification of the S(M,T) tier-routing math on Parallel.ai's
9-tier Task API pricing.

Score:  S(tier) = quality_prior(tier, class) * exp(-beta * cost_dollars(tier))

We verify four properties hold across difficulty classes and beta operating
points, then compute fleet-level savings for a mixed workload.
"""
import math

# Parallel.ai Task API list price -> dollars per single request (price_per_1k / 1000).
TIERS = {
    "Lite":    5    / 1000,   # 0.005
    "Base":    10   / 1000,   # 0.010
    "Core":    25   / 1000,   # 0.025
    "Core2x":  50   / 1000,   # 0.050
    "Pro":     100  / 1000,   # 0.100
    "Ultra":   300  / 1000,   # 0.300
    "Ultra2x": 600  / 1000,   # 0.600
    "Ultra4x": 1200 / 1000,   # 1.200
    "Ultra8x": 2400 / 1000,   # 2.400
}
ORDER = list(TIERS.keys())

# Quality prior = P(tier returns an acceptable answer | difficulty class).
# Easy queries saturate early (Lite already good); hard queries need depth.
SCENARIOS = {
    "A_easy   (HQ city of a known company)": {
        "Lite": 0.90, "Base": 0.95, "Core": 0.97, "Core2x": 0.98,
        "Pro": 0.99, "Ultra": 1.00, "Ultra2x": 1.00, "Ultra4x": 1.00, "Ultra8x": 1.00,
    },
    "B_medium (5 startups + funding + team)": {
        "Lite": 0.45, "Base": 0.62, "Core": 0.80, "Core2x": 0.85,
        "Pro": 0.92, "Ultra": 0.97, "Ultra2x": 0.98, "Ultra4x": 0.985, "Ultra8x": 0.99,
    },
    "C_hard   (full market teardown)": {
        "Lite": 0.20, "Base": 0.30, "Core": 0.55, "Core2x": 0.65,
        "Pro": 0.80, "Ultra": 0.92, "Ultra2x": 0.96, "Ultra4x": 0.98, "Ultra8x": 0.99,
    },
}

BETAS = [0.4, 4.0, 8.0, 20.0]


def score(quality, cost, beta):
    return quality * math.exp(-beta * cost)


def winner(priors, beta):
    best_tier, best_s = None, -1.0
    for t in ORDER:
        s = score(priors[t], TIERS[t], beta)
        if s > best_s:
            best_s, best_tier = s, t
    return best_tier, best_s


def print_score_table(name, priors):
    print(f"\n### {name}")
    header = "tier     cost$   q    " + "  ".join(f"b={b:<4g}" for b in BETAS)
    print(header)
    print("-" * len(header))
    for t in ORDER:
        row = f"{t:<8} {TIERS[t]:<6.3f} {priors[t]:.2f}  "
        cells = []
        for b in BETAS:
            cells.append(f"{score(priors[t], TIERS[t], b):.3f} ")
        print(row + " ".join(cells))
    print("winner:  " + "        " + "        ".join(winner(priors, b)[0] for b in BETAS))


def verify_properties():
    print("\n========== PROPERTY VERIFICATION ==========")
    ok = True

    # P1: beta -> 0 selects the maximum-quality tier (cost is ignored).
    print("\n[P1] beta->0  =>  argmax = highest-quality tier")
    for name, priors in SCENARIOS.items():
        w, _ = winner(priors, 1e-9)
        max_q_tiers = [t for t in ORDER if priors[t] == max(priors.values())]
        # the cheapest among the max-quality ties wins on the tie-break (first in ORDER)
        expect = next(t for t in ORDER if t in max_q_tiers)
        flag = "OK" if w == expect else "FAIL"
        ok &= (w == expect)
        print(f"   {name[:10]:<10} winner={w:<8} expect_max_q={expect:<8} {flag}")

    # P2: beta -> very large selects the cheapest tier (quality is dominated).
    print("\n[P2] beta->inf  =>  argmax = cheapest tier (Lite)")
    for name, priors in SCENARIOS.items():
        w, _ = winner(priors, 10000.0)
        flag = "OK" if w == "Lite" else "FAIL"
        ok &= (w == "Lite")
        print(f"   {name[:10]:<10} winner={w:<8} {flag}")

    # P3: monotone cost penalty. For two tiers of EQUAL quality, the cheaper
    #     one must score >= the dearer one at every beta>0.
    print("\n[P3] equal quality => cheaper tier never scores lower (any beta>0)")
    # construct a controlled equal-quality pair: Ultra vs Ultra8x at q=1.0
    for b in BETAS:
        s_cheap = score(1.0, TIERS["Ultra"], b)
        s_dear = score(1.0, TIERS["Ultra8x"], b)
        flag = "OK" if s_cheap >= s_dear else "FAIL"
        ok &= (s_cheap >= s_dear)
        print(f"   beta={b:<5g} Ultra={s_cheap:.4f} >= Ultra8x={s_dear:.4f} {flag}")

    # P4: difficulty monotonicity. At a fixed mid beta, the selected tier for
    #     hard >= medium >= easy in price (harder work buys up the ladder).
    print("\n[P4] at fixed beta=8, harder class selects a tier at least as expensive")
    b = 8.0
    we = TIERS[winner(SCENARIOS["A_easy   (HQ city of a known company)"], b)[0]]
    wm = TIERS[winner(SCENARIOS["B_medium (5 startups + funding + team)"], b)[0]]
    wh = TIERS[winner(SCENARIOS["C_hard   (full market teardown)"], b)[0]]
    flag = "OK" if we <= wm <= wh else "FAIL"
    ok &= (we <= wm <= wh)
    print(f"   easy=${we:.3f}  medium=${wm:.3f}  hard=${wh:.3f}  {flag}")

    print("\n========== ALL PROPERTIES " + ("PASS" if ok else "FAILED") + " ==========")
    return ok


MIX = {"A_easy   (HQ city of a known company)": 0.60,
       "B_medium (5 startups + funding + team)": 0.30,
       "C_hard   (full market teardown)": 0.10}


def fleet_point(beta):
    routed_cost = routed_quality = 0.0
    picks = {}
    for name, frac in MIX.items():
        priors = SCENARIOS[name]
        w, _ = winner(priors, beta)
        picks[name[:8]] = w
        routed_cost += frac * TIERS[w]
        routed_quality += frac * priors[w]
    always_ultra = sum(frac * TIERS["Ultra"] for frac in MIX.values())
    always_8x = sum(frac * TIERS["Ultra8x"] for frac in MIX.values())
    q_ultra = sum(frac * SCENARIOS[n]["Ultra"] for n, frac in MIX.items())
    return routed_cost, routed_quality, always_ultra, always_8x, q_ultra, picks


def fleet_savings():
    print("\n========== FLEET FRONTIER (mixed workload: 60% easy / 30% med / 10% hard) ==========")
    print("\n beta   routed$   save_vs_Ultra  qual   dQ_vs_Ultra   picks(easy/med/hard)")
    print("-" * 78)
    for b in [0.2, 0.4, 1.0, 2.0, 4.0, 8.0, 20.0]:
        rc, rq, au, a8, qu, picks = fleet_point(b)
        save = 100 * (au - rc) / au
        dq = 100 * (rq - qu)
        p = "/".join(picks[k] for k in ["A_easy  ", "B_medium", "C_hard  "])
        print(f" {b:<5g}  ${rc:.4f}   {save:5.1f}%        {rq:.3f}  {dq:+5.1f} pp     {p}")
    print("\n always-Ultra baseline: cost $0.3000/req, quality 0.983")
    print(" Read the knee: the operating point that holds quality within ~2pp of")
    print(" always-Ultra while still cutting cost is what the online learner targets.")


if __name__ == "__main__":
    print("PARALLEL.AI TIER ROUTING - S(M,T) MATH VERIFICATION")
    print("=" * 52)
    for name, priors in SCENARIOS.items():
        print_score_table(name, priors)
    verify_properties()
    fleet_savings()
