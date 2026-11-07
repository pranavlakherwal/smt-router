# S(M,T): Scoring AI Endpoints and the Measurement Gap in LLM Routing

**One equation decides which AI endpoint handles a given task.**

S(M,T) scores any endpoint (LLM, agent, script, or tool) against any task by multiplying three things. Does it meet the hard requirements? How capable is it across multiple dimensions? Is it worth the cost for this specific task? If any dealbreaker fails, the score is zero.

```
S(M,T) = [Gates] × [Compatibility] × [Cost Penalty]
```

That's it. Gates filter out endpoints that can't handle the task. Compatibility measures fit across 16 learned dimensions. Cost adjusts for what the task is worth. The highest-scoring endpoint wins. Set an accuracy bar and the bar becomes a gate: the cheapest endpoint that cleared it wins.

**[Download the paper (PDF, v1.5, 7 November 2026)](SMT_Router_Paper_v1.5.pdf)**. Previous: [v1.4, June 2026](SMT_Router_Paper_v1.4.pdf).

## Try it in a minute

```
git clone https://github.com/pranavlakherwal/smt-router.git
cd smt-router
pip install "pydantic>=2"
python -m router options --kind maths
```

```
Maths word problems, measured 2026-11-06
  90: inclusionai/ling-3.0-tiny@think-off  91 of 100, then 90 of 100 on fresh tasks  $0.01 per 1,000 tasks
  95: inclusionai/ling-3.0-flash  99 of 100, then 98 of 100 on fresh tasks  $0.02 per 1,000 tasks
  98: inclusionai/ling-3.0-flash  99 of 100, then 98 of 100 on fresh tasks  $0.02 per 1,000 tasks
```

No key, no network, no model call. The answer comes from a table we measured and dated.

## Which cheap model does this job well enough? (6 November 2026)

Routing should find the cheapest model that does the job, not the best model. So for each kind of job we name up to three options: the cheapest setting that got 90, 95 and 98 answers right in 100. Each one cleared its bar twice, on 100 tasks and again on 100 fresh ones. Prices are dollars per 1,000 tasks.

<!-- options -->
| Kind of job | Cheapest at 90 | Cheapest at 95 | Cheapest at 98 |
|---|---|---|---|
| Maths word problems | `inclusionai/ling-3.0-tiny@think-off`, $0.008 | `inclusionai/ling-3.0-flash`, $0.020 | `inclusionai/ling-3.0-flash`, $0.020 |
| Science questions | `inclusionai/ling-3.0-tiny@think-off`, $0.007 | `upstage/solar-mini4`, $0.017 | `openai/gpt-6-luna`, $0.043 |
| Checking a claim against a paper | none | none | none |
| General knowledge exam | `inception/mercury-2.5@think-off`, $0.031 | none | none |
<!-- /options -->

56 models in 66 settings answered the same 400 tasks on 5 and 6 November 2026. We measured this ourselves, and no outside party has checked it. Three things to know before you use it:

1. **One good run is not enough.** Of 21 hits we re-tested on fresh tasks, 7 fell under their bar.
2. **Two of the four task sets cannot show a 95.** Their answer keys look wrong on 5 to 7 tasks in 100.
3. **Claim checking has no option yet.** The three cheapest hits fell to about 81 on fresh tasks.

The method, every table and the limits are in [docs/ROUTE_OPTIONS_2026-11.md](docs/ROUTE_OPTIONS_2026-11.md). `python -m router report` recounts every table from the files in `data/`.

## Why This Matters

The world is headed toward spending billions on AI tokens every year. Most routing today is manual or nonexistent. Every query gets sent to the same model, regardless of difficulty. A formatting task and a complex reasoning problem get the same $0.06 API call.

S(M,T) routes intelligently. In the paper's benchmark replay it retained 94.35% of best-model quality at half the cost (offline, public benchmarks). Routing of this kind can cut AI spend by up to about 80% while keeping most of the quality. That 80% is arithmetic from the same RouteLLM table (paper v1.5, Section 8.5.2, Table 9), not a measurement of spend. Route 20% of queries to the strong model and 80% to the cheap one, and the learned router kept 89.02% of the strong model's quality. At 50% it kept 94.35%. Both numbers date from the paper's first version, 25 March 2026; the arithmetic was confirmed run on 7 November 2026. On live models in November 2026 the picture is narrower; it is dated in the sections above and below.

## Key Results (paper v1.5, 7 November 2026; offline public benchmarks, first reported 25 March 2026, not re-run since)

| Benchmark | Result |
|-----------|--------|
| RouterBench accuracy | 83.63% (vs. 85.51% always-strong, 96.45% oracle) |
| RouterBench 5-shot | 86.76% (beats always-strong 86.66%) |
| RouteLLM AUC | 0.8006 |
| Quality at 50% cost | 94.35% of always-strong |
| Cross-benchmark | One model generalizes across both benchmarks without tuning |

## What's Different

**Heterogeneous endpoints.** Every other router assumes all endpoints are LLMs. S(M,T) scores LLMs, agents, scripts, and tool-use systems through the same equation. A script has no context window. An agent has no per-token cost. The gate layer handles type-specific constraints so the scoring equation doesn't have to.

**The measurement gap.** We tried 14 hand-designed scoring functions on 2.76M benchmark records. All 14 failed. Benchmarks use incomparable scales, don't transfer to deployment, and collapse complex properties into single numbers. This is documented in detail in the paper. It's the finding we think matters most.

**Learned scoring that generalizes.** Instead of hand-designing what "reasoning ability" means as a number, we trained 16 bilinear interaction matrices on 740K routing examples. The model generalizes across benchmarks it wasn't trained for. When we removed an entire data source and retrained, it still worked on the missing source.

**Scaling didn't help.** A 4.6x bigger model got worse on every metric. The bottleneck is data quality, not model capacity.

## Paper

The paper (v1.5, 7 November 2026) covers:

- The S(M,T) equation and why it's designed this way
- The measurement gap: why hand-designed scoring functions fail structurally
- Learned bilinear scoring (3.15M parameters, 740K training samples)
- Cross-benchmark generalization
- Why scaling failed and what that means
- Self-critique: what's genuinely new vs. standard techniques applied to a new problem
- The November 2026 measurement on live models: 66 settings, four kinds of job (Section 9.10)

Convergence proofs and adversarial analysis are in the appendices.

> v1.5 (7 November 2026) resolves SMT-ERR-004 and SMT-ERR-005. It states that the production cost cut was reported with no measure of answer quality. It adds the accuracy bar as a sixth gate and reports the November 2026 measurement on live models. v1.4 (June 2026) corrected the framing of the core equation. Under a log transform and softmax it is a constrained conditional logit (McFadden, 1974) with a log-sum compatibility term. The contribution is the parameterization for heterogeneous endpoint routing, its identifiability, convergence and regret analysis, and the production results. See paper/MCFADDEN_RESOLUTION.md and paper/ERRATA_QUEUE.md. See [CHANGELOG.md](CHANGELOG.md) for details. Previous version: [v1.0 release](https://github.com/pranavlakherwal/smt-router/releases/tag/v1.0).

## Code

The routing engine needs Python 3.9 or later and `pydantic`. It reads two small files in `data/` and calls no model.

```
pip install "pydantic>=2"
python examples/route_one.py        # one task, routed two ways, with the reasons
python -m router options            # the three options for every measured kind of job
python -m router route --kind maths --bar 95 "<task>"    # the recommended call
python -m router route --kind science --bar 95 "Why does ice float on water?"
python -m router table --kind exam --top 10
pip install pytest && python -m pytest -q
```

95 is the bar we use ourselves by default, decided 7 November 2026. Where no setting cleared 95 twice the command says so.

```python
from router.engine import RoutingEngine
from router.models import Task, TaskType

engine = RoutingEngine()

# The cheapest endpoint that cleared 90 right in 100, twice, on this kind of job.
decision = engine.route(Task(query="A shop sells pens at 3 for 2 dollars. What do 12 cost?", kind="maths", bar=90))

# The highest S(M,T) score, as in the paper. No bar.
decision = engine.route(Task(query="A shop sells pens at 3 for 2 dollars. What do 12 cost?", task_type=TaskType.MATH))

print(engine.explain(decision))
```

What the engine does today:

- **With a bar,** the bar is a gate. An endpoint passes only if it cleared the bar twice on measured tasks of that kind. The cheapest endpoint that passes is selected. This is the rule the November measurements support.
- **With no bar,** the engine ranks by the paper's score, with the measured pass rate as the capability input. That ranking picks the most accurate cheap setting. We have not shown that it beats a simpler rule.
- **It refuses instead of guessing.** An empty query is an error. A misnamed field is an error. A bar on a kind of job with no measured tasks selects nothing and says "NOT MEASURED".
- **It writes nothing** unless you set `SMT_ROUTER_STATE_DIR`. See `.env.example`.

The benchmark database behind the paper's data-grounded scores (14.3 million rows) is not shipped. Without it those scores are fixed priors, and every score says where it came from: measured, computed, runtime or prior.

| File | Description |
|------|-------------|
| `engine.py` | Main routing engine |
| `measured.py` | The measured table: bars, options, endpoints |
| `gates.py` | Multiplicative gate layer, six gates |
| `scorer.py` | S(M,T) scoring implementation |
| `phi_evaluators.py`, `phi_mapper.py` | The sixteen phi functions and the records that ground them |
| `registry.py`, `models.py`, `config.py` | Endpoints, data models, paths and settings |
| `report.py`, `__main__.py` | The research tables and the command line |
| `weight_learner.py` | Online weight learning (Robbins-Monro) |
| `learned_phi.py` | BilinearPhiEngine (16 learned bilinear heads) |
| `train_bilinear.py` | Training script |
| `eval_bilinear.py` | Evaluation on RouterBench and RouteLLM |
| `data_pipeline.py` | Unified data ingestion from 5 public datasets |
| `leave_one_out.py` | Leave-one-out generalization analysis |
| `model_registry.py` | Model registry (180 aliases to 48 models) |

The learned scorer and its training and evaluation scripts need PyTorch and the rest of the `learned` extra in `pyproject.toml`. They have not been run on the November tasks.

| Data file | What it holds |
|------|-------------|
| `data/measured_capability.json` | Right answers out of 100 per setting and kind, measured cost, list price, date |
| `data/outcomes_2026-11-06.json` | Right or wrong for every setting on every task |
| `data/suspect_keys_2026-11-06.json` | Answer keys the strongest settings disagree with |
| `data/seed_weights.json` | Seed weights of the sixteen phi functions |

## Training Data

| Source | Samples | Models | Score Type |
|--------|---------|--------|------------|
| RouterBench | 401,467 | 11 | Accuracy (0/1) |
| RouteLLM | 218,202 | 2 | Win rate (0/1) |
| Arena 55K | 111,712 | 45 | Elo-derived (continuous) |
| MT-Bench | 6,710 | 5 | Rating (1-10, normalized) |
| RewardBench | 2,712 | 31 | Accuracy (0/1) |
| **Total** | **740,803** | **48** | |

## What we can say about reliability (6 November 2026)

The benchmark table above is from the paper (first reported 25 March 2026, unchanged through v1.5): offline replays of public benchmarks, run once by us and not re-run since. Everything in this section is measured on our own systems, with the date, and it is labelled directional. No outside party has checked it.

| Measure | Value | Date | Note |
|---|---|---|---|
| Cost in offline replay, 1,622 real routing decisions | $101.30 to $49.00, a 51.6% cut | 25 May 2026 | cost only. Quality was not measured in that replay |
| Recommended route correct, 200 tasks in four task sets, five routes, every route on every task | 171 of 200, 86% (95% interval 80% to 90%) | 5 Nov 2026 | best single route 174 of 200. Empty replies at a 1,000-token cap were asked again at 4,000 |
| Same, at a 1,000-token answer budget (the pre-registered reading) | 146 of 200, 73% (67% to 79%) | 5 Nov 2026 | the recommended route gave no answer on 46 tasks. Best single route 163 |
| Does the best route change with the task set? | no, p = 0.20 once empty replies were re-asked | 5 Nov 2026 | at the 1,000-token cap it looked like yes, p = 0.0005. The cap produced that signal |
| Recommended route matches the best observed route, 20 SciFact tasks with repeated runs | 3 of 20 (5% to 36%) | 4 Nov 2026 | one fixed model was recommended on all 20 |
| Kinds of job with a cheapest setting that cleared its bar twice, of 12 bars across four kinds | 7 of 12 | 6 Nov 2026 | 66 settings, 100 tasks then 100 fresh tasks per kind. Maths 3, science 3, exam 1, claims 0 |
| Hits that fell under their bar on fresh tasks | 7 of 21 | 6 Nov 2026 | a hit is a setting that reached a bar on the first 100 tasks |
| Answer keys the strongest settings disagree with, per 100 tasks | maths 0, science 1, claims 7, exam 5 | 6 Nov 2026 | at least four of the five strongest settings agree against the key |

Task sets: GSM8K, MMLU, ARC-Challenge and SciFact, 50 tasks each, drawn with a fixed seed. Checkers hold no model. The question, the sets, the routes and the statistics were written down before the first call.

What these rows measure: the shadow router that runs on our own prompts, which scores capability times complexity times cost from a static registry. They do not measure the learned bilinear scorer in `router/learned_phi.py`, which has not been run on these tasks.

What we read from them:

1. Given room to answer, the recommendation is about as accurate as always using the best single model. It neither helps nor hurts by a margin visible at 200 tasks.
2. The router has no idea of an answer budget. At a tight budget it recommends reasoning-heavy models that run out of tokens, and two cheaper fixed routes beat it.
3. Among five low-cost models on four task sets, picking a route per task set bought 4 tasks in 200. Cost is where the routes differ: the most accurate route cost more than twenty times the cheapest for 16 more correct answers in 200.

4. Capability has to be measured per kind of job. Before the 6 November run our router rated most models from a leaderboard row or a guess from price. It rated one model 0.24 for maths, and that model got 99 of 100.
5. The score in the paper picks the most accurate setting. The cheapest setting that does the job is a different pick, and on the exam it cost a third as much. The engine here now treats the accuracy bar as a gate and picks on measured cost.

So the honest recommendation text is: S(M,T) is a scoring form with a cost replay and a measured table behind it, not a validated router. Use the table to pick a cheap model for maths word problems, science questions or an exam-style job, at the bar you need. Re-test on your own tasks before you switch live traffic.

The shadow router that logs a pick beside every prompt without changing it runs inside our own systems today (19,000 logged decisions since 20 March 2026). It is not yet packaged for release. Since 6 November it also logs the three options beside its pick.

## Requirements

- The routing engine, the options and the tables: Python 3.9 or later and pydantic 2
- The learned scorer and its scripts: PyTorch, sentence-transformers (all-mpnet-base-v2 for prompt encoding), numpy, pandas, scipy

## Citation

```bibtex
@article{lakherwal2026smt,
  title={S(M,T): Scoring AI Endpoints and the Measurement Gap in LLM Routing},
  author={Lakherwal, Pranav},
  year={2026},
  month={March}
}
```

## Looking for arXiv Endorsement

I'm looking for a collaborator with arXiv endorsement in cs.AI, cs.LG, or cs.CL to help submit this paper. If you have endorsement privileges and find this work interesting, please reach out: pranav@pranavlakherwal.com

## License

MIT License. See [LICENSE](LICENSE) for details.

## AI Disclosure

Developed with AI assistance from Anthropic. All research direction, experimental design, and intellectual contributions are the author's own.

## Author

Pranav Lakherwal (pranav@pranavlakherwal.com)
