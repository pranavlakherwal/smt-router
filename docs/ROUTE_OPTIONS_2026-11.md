# Three options per kind of job: the cheapest model that clears 90, 95 and 98

For each kind of job we name the cheapest model setting that got 90, 95 and 98 answers right in 100, twice.

Written 6 November 2026. Measured by us on 5 and 6 November 2026. No outside party has checked it. Every table below is printed from the files in `data/` by `python -m router report`.

## The rule

Routing should find the cheapest model that does the job, not the best model.

1. A setting **hits** a bar when its right answers on 100 tasks reach the bar.
2. It **holds** the bar when it reaches it again on 100 fresh tasks.
3. The **option** at a bar is the cheapest setting that hit and held, by measured cost per task.

A setting is a model with its request settings. An entry ending `@think-off` is the same model with thinking switched off.

## The options

<!-- t1 -->
| Kind of job | Bar | Cheapest setting that cleared it twice | First tasks | Fresh tasks | 95% interval, both runs | $ per 1,000 tasks |
|---|---|---|---|---|---|---|
| Maths word problems | 90 | `inclusionai/ling-3.0-tiny@think-off` | 91 of 100 | 90 of 100 | 86 to 94 | 0.008 |
| Maths word problems | 95 | `inclusionai/ling-3.0-flash` | 99 of 100 | 98 of 100 | 96 to 99 | 0.020 |
| Maths word problems | 98 | `inclusionai/ling-3.0-flash` | 99 of 100 | 98 of 100 | 96 to 99 | 0.020 |
| Science questions | 90 | `inclusionai/ling-3.0-tiny@think-off` | 94 of 100 | 91 of 100 | 88 to 95 | 0.007 |
| Science questions | 95 | `upstage/solar-mini4` | 96 of 100 | 95 of 100 | 92 to 98 | 0.017 |
| Science questions | 98 | `openai/gpt-6-luna` | 98 of 100 | 98 of 100 | 95 to 99 | 0.043 |
| Checking a claim against a paper | 90 | none |  |  |  |  |
| Checking a claim against a paper | 95 | none |  |  |  |  |
| Checking a claim against a paper | 98 | none |  |  |  |  |
| General knowledge exam | 90 | `inception/mercury-2.5@think-off` | 90 of 100 | 93 of 100 | 87 to 95 | 0.031 |
| General knowledge exam | 95 | none |  |  |  |  |
| General knowledge exam | 98 | none |  |  |  |  |
<!-- /t1 -->

The interval column pools both runs, 200 tasks. Read it before you trust a bar. A setting that scored 90 of 100 has a 95% interval of 83 to 94. "Cleared 90 twice" is a measured fact about 200 tasks. It is not a promise about the next 100.

## What we found

**1. A bar belongs to a kind of job.** `inclusionai/ling-3.0-flash` got 99 of 100 on maths and 85 of 100 on claim checking. One number per model hides that.

**2. One good run is not enough.** We re-tested 21 pairings of a setting and a kind on fresh tasks. Seven fell under the top bar they had hit. Table 3 lists all 21.

**3. A bar needs a test that can measure it.** No setting passed 94 on claim checking or on the exam. On 12 of those 200 tasks, at least four of the five strongest settings agree with each other and disagree with the answer key. Table 8 lists them. On `mmlu-3299`, 61 of 66 settings give one answer and the key gives another. A dearer model cannot fix a wrong key, so those two task sets cannot show a 95 or a 98.

**4. Claim checking has no option yet.** Eleven settings hit 90 on the first tasks. The three cheapest were re-tested and scored 81, 81 and 82. The first claim tasks are balanced by label. The fresh ones come from another split of the same data with its own label mix. We do not yet know which reading is the true one.

**5. The paper's score picks the most accurate setting, not the cheapest that does the job.** Table 6 feeds the measured pass rates into the score from the paper. On the exam it picks a setting at $0.097 per 1,000 tasks for 94 right. The option at 90 costs $0.031. The cost term in the score has five price tiers, and 61 of the 66 settings sit in one of them. So cost cannot separate them, and accuracy decides. The engine now treats the bar as a gate and picks on measured cost.

**6. Thinking off is a cheaper setting of the same model.** In ten pairs, the setting with thinking off cost about a third as much. It lost about 10 answers in 400 on average. In three of the ten pairs it lost none. Table 5 has every pair.

**7. Price and leaderboards were poor stand-ins for capability.** Before this run our own router rated each model from public leaderboards, or guessed from its price. On 6 November 2026 its list held 373 models: 35 with their own leaderboard row, 56 with a sibling's row, 282 with a guess from price. It rated `inclusionai/ling-3.0-flash` at 0.24 for maths because the model is cheap. That model got 99 of 100.

## How it was measured

| What | How |
|---|---|
| Kinds of job | Maths word problems (GSM8K, test split). Science questions (ARC-Challenge, test split). Checking a claim against a paper (SciFact). A general knowledge exam (MMLU, test split) |
| Tasks | 100 per kind, drawn with a fixed seed. Then 100 fresh tasks per kind, drawn with another seed and no overlap |
| Claim tasks | First tasks: validation split, balanced across the three labels. Fresh tasks: train split |
| Models | 56 models in 66 settings, called through OpenRouter on 5 and 6 November 2026 |
| Price range | List prices up to $0.75 per million input tokens. Dearer models were planned and not reached before the spend stop |
| Order | Cheapest first. A 20-task trial dropped settings that failed to answer. The rest answered all 400 first tasks |
| Checking | No model checks an answer. The final number, letter or label must match the key |
| Limits on a reply | 4,000 output tokens. An empty reply counts as wrong. One provider refusal counts as wrong |
| Re-test | For each kind and bar, the three cheapest settings that hit it answered the fresh tasks |
| Plan | Written and committed before the first paid call. The refusal rule was added after the rows existed |
| Cost | $3.62 of stored spend across both days, for about 28,600 stored replies |

## The tables

### Table 2. How many settings reached each bar

<!-- t2 -->
| Kind of job | Bar | Settings that hit it on the first tasks | Of those, re-tested on fresh tasks | Held it on fresh tasks |
|---|---|---|---|---|
| Maths word problems | 90 | 54 | 6 | 6 |
| Maths word problems | 95 | 37 | 5 | 5 |
| Maths word problems | 98 | 13 | 3 | 2 |
| Science questions | 90 | 58 | 9 | 8 |
| Science questions | 95 | 38 | 6 | 6 |
| Science questions | 98 | 13 | 3 | 1 |
| Checking a claim against a paper | 90 | 11 | 3 | 0 |
| Checking a claim against a paper | 95 | 0 | 0 | 0 |
| Checking a claim against a paper | 98 | 0 | 0 | 0 |
| General knowledge exam | 90 | 9 | 3 | 3 |
| General knowledge exam | 95 | 0 | 0 | 0 |
| General knowledge exam | 98 | 0 | 0 | 0 |
<!-- /t2 -->

### Table 3. Every hit that was re-tested on fresh tasks

<!-- t3 -->
| Kind of job | Setting | First tasks | Fresh tasks | Top bar hit | Top bar held | Result |
|---|---|---|---|---|---|---|
| Maths word problems | `inclusionai/ling-3.0-tiny@think-off` | 91 of 100 | 90 of 100 | 90 | 90 | held |
| Maths word problems | `inclusionai/ling-3.0-flash` | 99 of 100 | 98 of 100 | 98 | 98 | held |
| Maths word problems | `inception/mercury-2.5@think-off` | 95 of 100 | 99 of 100 | 95 | 98 | held |
| Maths word problems | `mistralai/mistral-small-24b-instruct-2501` | 97 of 100 | 96 of 100 | 95 | 95 | held |
| Maths word problems | `nex-agi/nex-n2.5-mini` | 98 of 100 | 97 of 100 | 98 | 95 | slipped |
| Maths word problems | `meta-llama/llama-4-scout` | 98 of 100 | 98 of 100 | 98 | 98 | held |
| Science questions | `inclusionai/ling-3.0-tiny@think-off` | 94 of 100 | 91 of 100 | 90 | 90 | held |
| Science questions | `nvidia/nemotron-3-nano-30b-a3b@think-off` | 91 of 100 | 92 of 100 | 90 | 90 | held |
| Science questions | `mistralai/ministral-3b-2512` | 92 of 100 | 85 of 100 | 90 | none | slipped |
| Science questions | `upstage/solar-mini4` | 96 of 100 | 95 of 100 | 95 | 95 | held |
| Science questions | `nex-agi/nex-n2.5-mini` | 97 of 100 | 97 of 100 | 95 | 95 | held |
| Science questions | `mistralai/mistral-small-24b-instruct-2501` | 95 of 100 | 96 of 100 | 95 | 95 | held |
| Science questions | `inclusionai/ling-3.0-flash` | 98 of 100 | 96 of 100 | 98 | 95 | slipped |
| Science questions | `google/gemma-4-26b-a4b-it` | 98 of 100 | 95 of 100 | 98 | 95 | slipped |
| Science questions | `openai/gpt-6-luna` | 98 of 100 | 98 of 100 | 98 | 98 | held |
| Checking a claim against a paper | `deepseek/deepseek-v4-flash` | 91 of 100 | 81 of 100 | 90 | none | slipped |
| Checking a claim against a paper | `google/gemma-4-31b-it` | 94 of 100 | 81 of 100 | 90 | none | slipped |
| Checking a claim against a paper | `qwen/qwen3.7-flash` | 92 of 100 | 82 of 100 | 90 | none | slipped |
| General knowledge exam | `inception/mercury-2.5@think-off` | 90 of 100 | 93 of 100 | 90 | 90 | held |
| General knowledge exam | `openai/gpt-6-luna` | 94 of 100 | 97 of 100 | 90 | 95 | held |
| General knowledge exam | `google/gemma-4-31b-it` | 91 of 100 | 97 of 100 | 90 | 95 | held |
<!-- /t3 -->

### Table 4. Every setting on the first 400 tasks

<!-- t4 -->
| Setting | Maths word problems | Science questions | Checking a claim against a paper | General knowledge exam | Right of 400 | $ for the 400 tasks | Empty replies | Cut at the limit |
|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-31b-it` | 98 | 98 | 94 | 91 | 381 | 0.0311 | 0 | 0 |
| `google/gemini-3.7-flash` | 98 | 98 | 92 | 93 | 381 | 0.4539 | 0 | 0 |
| `inception/mercury-2.5` | 99 | 98 | 91 | 89 | 377 | 0.0585 | 0 | 0 |
| `inclusionai/ling-3.1-flash` | 99 | 98 | 92 | 86 | 375 | 0.0471 | 9 | 10 |
| `openai/gpt-5.6-luna` | 95 | 98 | 90 | 92 | 375 | 0.0612 | 0 | 0 |
| `z-ai/glm-5.3-flash` | 98 | 97 | 92 | 87 | 374 | 0.0436 | 6 | 6 |
| `qwen/qwen3.7-flash@think-off` | 97 | 98 | 89 | 89 | 373 | 0.0175 | 0 | 1 |
| `openai/gpt-6-luna` | 95 | 98 | 86 | 94 | 373 | 0.0307 | 0 | 0 |
| `xiaomi/mimo-v2.6-flash` | 96 | 98 | 91 | 88 | 373 | 0.0349 | 1 | 1 |
| `inception/mercury-3` | 95 | 98 | 90 | 90 | 373 | 0.0510 | 6 | 6 |
| `deepseek/deepseek-v4-flash` | 96 | 97 | 91 | 88 | 372 | 0.0235 | 7 | 6 |
| `deepseek/deepseek-v4.1-flash` | 97 | 98 | 87 | 90 | 372 | 0.0831 | 5 | 5 |
| `inclusionai/ling-3.0-flash` | 99 | 98 | 85 | 89 | 371 | 0.0106 | 5 | 5 |
| `google/gemma-4-26b-a4b-it` | 97 | 98 | 87 | 89 | 371 | 0.0211 | 0 | 1 |
| `upstage/solar-pro4` | 98 | 98 | 87 | 88 | 371 | 0.0475 | 0 | 0 |
| `nex-agi/nex-n2.5-mini` | 98 | 97 | 87 | 88 | 370 | 0.0116 | 6 | 6 |
| `deepseek/deepseek-v3.2` | 97 | 96 | 87 | 90 | 370 | 0.0417 | 0 | 0 |
| `tencent/hy4@think-off` | 97 | 97 | 85 | 91 | 370 | 0.0887 | 0 | 0 |
| `qwen/qwen3.8-flash` | 99 | 96 | 91 | 84 | 370 | 0.0889 | 13 | 13 |
| `bytedance-seed/seed-2.0-mini@think-off` | 97 | 97 | 89 | 86 | 369 | 0.0379 | 0 | 1 |
| `nex-agi/nex-n2.5-pro` | 95 | 97 | 87 | 89 | 368 | 0.0337 | 6 | 6 |
| `qwen/qwen3.7-flash` | 99 | 96 | 92 | 81 | 368 | 0.0573 | 12 | 12 |
| `minimax/minimax-m3.1-flash` | 97 | 96 | 87 | 85 | 365 | 0.0941 | 9 | 8 |
| `meta-llama/llama-4-scout` | 98 | 96 | 82 | 87 | 363 | 0.0397 | 0 | 0 |
| `bytedance-seed/seed-2.0-mini` | 98 | 97 | 81 | 87 | 363 | 0.1120 | 9 | 9 |
| `minimax/minimax-m2.5` | 95 | 97 | 88 | 83 | 363 | 0.3575 | 9 | 8 |
| `openai/gpt-oss-120b` | 97 | 97 | 86 | 82 | 362 | 0.0327 | 0 | 0 |
| `bytedance-seed/seed-1.6-flash` | 98 | 97 | 86 | 81 | 362 | 0.0910 | 5 | 5 |
| `inception/mercury-2.5@think-off` | 95 | 93 | 83 | 90 | 361 | 0.0097 | 0 | 0 |
| `mistralai/mistral-small-24b-instruct-2501` | 97 | 95 | 82 | 85 | 359 | 0.0100 | 0 | 0 |
| `bytedance-seed/seed-1.6-flash@think-off` | 97 | 97 | 79 | 84 | 357 | 0.0411 | 0 | 0 |
| `meta-llama/llama-3.3-70b-instruct` | 96 | 95 | 84 | 82 | 357 | 0.0700 | 0 | 0 |
| `mistralai/mistral-small-2603` | 93 | 97 | 81 | 85 | 356 | 0.0326 | 0 | 0 |
| `microsoft/phi-4` | 93 | 95 | 85 | 82 | 355 | 0.0132 | 0 | 0 |
| `nvidia/nemotron-3-nano-30b-a3b` | 86 | 96 | 88 | 84 | 354 | 0.0608 | 19 | 19 |
| `z-ai/glm-4.7-flash` | 92 | 93 | 86 | 82 | 353 | 0.1535 | 24 | 24 |
| `mistralai/mistral-small-3.2-24b-instruct` | 95 | 96 | 76 | 85 | 352 | 0.0161 | 0 | 0 |
| `qwen/qwen3-30b-a3b-instruct-2507` | 94 | 96 | 77 | 85 | 352 | 0.0223 | 0 | 1 |
| `mistralai/ministral-14b-2512` | 96 | 95 | 80 | 79 | 350 | 0.0218 | 0 | 0 |
| `openai/gpt-oss-20b` | 94 | 93 | 83 | 79 | 349 | 0.0191 | 6 | 6 |
| `google/gemma-3-27b-it` | 96 | 94 | 79 | 80 | 349 | 0.0239 | 0 | 1 |
| `stepfun/step-3.5-flash` | 92 | 94 | 84 | 79 | 349 | 0.1247 | 28 | 28 |
| `prism-ml/ternary-bonsai-2-9b@think-off` | 95 | 92 | 81 | 79 | 347 | 0.0238 | 0 | 2 |
| `ibm-granite/granite-4.2-8b` | 85 | 94 | 86 | 81 | 346 | 0.0828 | 29 | 31 |
| `prism-ml/ternary-bonsai-2-9b` | 89 | 94 | 87 | 75 | 345 | 0.0764 | 29 | 31 |
| `upstage/solar-mini4` | 93 | 96 | 73 | 82 | 344 | 0.0128 | 0 | 2 |
| `openai/gpt-4o-mini` | 92 | 94 | 80 | 78 | 344 | 0.0311 | 0 | 0 |
| `mistralai/ministral-8b-2512` | 95 | 94 | 75 | 78 | 342 | 0.0160 | 0 | 1 |
| `z-ai/glm-4.7-flash@think-off` | 91 | 94 | 77 | 78 | 340 | 0.0243 | 0 | 1 |
| `ibm-granite/granite-4.2-8b@think-off` | 93 | 92 | 73 | 82 | 340 | 0.0287 | 0 | 11 |
| `amazon/nova-lite-v1` | 87 | 92 | 81 | 79 | 339 | 0.0156 | 0 | 0 |
| `openai/gpt-5.4-nano` | 88 | 89 | 80 | 79 | 336 | 0.0433 | 0 | 0 |
| `inclusionai/ling-3.0-tiny` | 89 | 87 | 85 | 73 | 334 | 0.0163 | 0 | 36 |
| `nvidia/nemotron-3.5-lightning` | 92 | 90 | 74 | 77 | 333 | 0.0306 | 6 | 8 |
| `mistralai/ministral-3b-2512` | 91 | 92 | 74 | 74 | 331 | 0.0110 | 0 | 1 |
| `google/gemma-3-12b-it` | 92 | 90 | 71 | 74 | 327 | 0.0162 | 0 | 0 |
| `openai/gpt-4.1-nano` | 90 | 90 | 68 | 79 | 327 | 0.0228 | 0 | 0 |
| `nvidia/nemotron-3.5-lightning@think-off` | 90 | 92 | 67 | 74 | 323 | 0.0127 | 0 | 4 |
| `inclusionai/ling-3.0-tiny@think-off` | 91 | 94 | 59 | 74 | 318 | 0.0038 | 0 | 2 |
| `google/gemini-2.5-flash-lite` | 92 | 80 | 80 | 66 | 318 | 0.0447 | 0 | 1 |
| `nvidia/nemotron-3-nano-30b-a3b@think-off` | 88 | 91 | 53 | 79 | 311 | 0.0096 | 0 | 0 |
| `mistralai/mistral-nemo` | 84 | 86 | 69 | 63 | 302 | 0.0032 | 0 | 0 |
| `google/gemma-3-4b-it` | 83 | 81 | 60 | 65 | 289 | 0.0116 | 0 | 1 |
| `meta-llama/llama-3.1-8b-instruct` | 84 | 76 | 68 | 59 | 287 | 0.0058 | 0 | 3 |
| `amazon/nova-micro-v1` | 41 | 87 | 72 | 63 | 263 | 0.0081 | 0 | 0 |
| `meta-llama/llama-3.2-1b-instruct` | 33 | 0 | 29 | 1 | 63 | 0.0166 | 0 | 0 |
<!-- /t4 -->

### Table 5. The same model with thinking on and off

<!-- t5 -->
| Model | Right of 400, thinking on | Thinking off | Change | $ for 400, on | $ for 400, off | Off cost as a share of on |
|---|---|---|---|---|---|---|
| `bytedance-seed/seed-1.6-flash` | 362 | 357 | -5 | 0.0910 | 0.0411 | 0.45 |
| `bytedance-seed/seed-2.0-mini` | 363 | 369 | 6 | 0.1120 | 0.0379 | 0.34 |
| `ibm-granite/granite-4.2-8b` | 346 | 340 | -6 | 0.0828 | 0.0287 | 0.35 |
| `inception/mercury-2.5` | 377 | 361 | -16 | 0.0585 | 0.0097 | 0.17 |
| `inclusionai/ling-3.0-tiny` | 334 | 318 | -16 | 0.0163 | 0.0038 | 0.23 |
| `nvidia/nemotron-3-nano-30b-a3b` | 354 | 311 | -43 | 0.0608 | 0.0096 | 0.16 |
| `nvidia/nemotron-3.5-lightning` | 333 | 323 | -10 | 0.0306 | 0.0127 | 0.41 |
| `prism-ml/ternary-bonsai-2-9b` | 345 | 347 | 2 | 0.0764 | 0.0238 | 0.31 |
| `qwen/qwen3.7-flash` | 368 | 373 | 5 | 0.0573 | 0.0175 | 0.30 |
| `z-ai/glm-4.7-flash` | 353 | 340 | -13 | 0.1535 | 0.0243 | 0.16 |
<!-- /t5 -->

### Table 6. What the paper's score picks when given the measured table

The pick is scored on the same tasks that set its pass rate, so this table flatters it.

<!-- t6 -->
| Kind of job | The score's pick, no bar | First tasks | $ per 1,000 tasks, first tasks | Settings tied on the top score | The option at 90: setting, first tasks, $ per 1,000 |
|---|---|---|---|---|---|
| Maths word problems | `inclusionai/ling-3.0-flash` | 99 of 100 | 0.020 | 5 | `inclusionai/ling-3.0-tiny@think-off`, 91 of 100, $0.008 |
| Science questions | `inclusionai/ling-3.0-flash` | 98 of 100 | 0.021 | 11 | `inclusionai/ling-3.0-tiny@think-off`, 94 of 100, $0.007 |
| Checking a claim against a paper | `google/gemma-4-31b-it` | 94 of 100 | 0.060 | 1 | none |
| General knowledge exam | `openai/gpt-6-luna` | 94 of 100 | 0.097 | 1 | `inception/mercury-2.5@think-off`, 90 of 100, $0.031 |
<!-- /t6 -->

### Table 7. Tasks that almost no setting gets right

<!-- t7 -->
| Kind of job | Best score of any setting, of 100 | Tasks all five strongest settings got wrong | Tasks nine in ten settings got wrong |
|---|---|---|---|
| Maths word problems | 99 | 0 | 0 |
| Science questions | 98 | 2 | 1 |
| Checking a claim against a paper | 94 | 5 | 3 |
| General knowledge exam | 94 | 4 | 4 |
<!-- /t7 -->

### Table 8. Answer keys the strongest settings disagree with

The rule: at least four of the five strongest settings on a kind give the same answer, and the key says otherwise. Open a task in the public data set by its id to read the question.

<!-- t8 -->
| Kind of job | Task | The key says | The strongest settings say | Settings that say the same | Settings that match the key |
|---|---|---|---|---|---|
| Maths word problems | none |  |  |  |  |
| Science questions | `arc-931` | B | C | 65 of 66 | 0 of 66 |
| Checking a claim against a paper | `scifact-sup-024` | SUPPORT | REFUTE | 50 of 66 | 8 of 66 |
| Checking a claim against a paper | `scifact-sup-028` | SUPPORT | NOINFO | 33 of 66 | 31 of 66 |
| Checking a claim against a paper | `scifact-sup-034` | SUPPORT | NOINFO | 64 of 66 | 2 of 66 |
| Checking a claim against a paper | `scifact-noi-031` | NOINFO | REFUTE | 50 of 66 | 0 of 66 |
| Checking a claim against a paper | `scifact-noi-034` | NOINFO | SUPPORT | 66 of 66 | 0 of 66 |
| Checking a claim against a paper | `scifact-ref-036` | REFUTE | SUPPORT | 30 of 66 | 18 of 66 |
| Checking a claim against a paper | `scifact-ref-037` | REFUTE | NOINFO | 47 of 66 | 13 of 66 |
| General knowledge exam | `mmlu-3299` | D | C | 61 of 66 | 0 of 66 |
| General knowledge exam | `mmlu-10177` | C | A | 60 of 66 | 2 of 66 |
| General knowledge exam | `mmlu-11334` | C | D | 57 of 66 | 0 of 66 |
| General knowledge exam | `mmlu-3440` | D | B | 54 of 66 | 2 of 66 |
| General knowledge exam | `mmlu-8260` | B | C | 45 of 66 | 14 of 66 |
<!-- /t8 -->

## Limits

- **Small samples.** 100 tasks per kind gives wide intervals. The gap between 95 and 98 is three tasks.
- **Public tasks.** These task sets are public and old. A model may have seen them in training, so a pass rate here is likely higher than on new work of the same kind.
- **Four kinds only.** No coding, writing or long documents were measured. The engine refuses to name an option for a kind it has no tasks for.
- **The cheap end only.** No model above $0.75 per million input tokens was measured. That leaves out every Anthropic model and the large models of OpenAI and Google.
- **One day, one route.** Prices and providers move. OpenRouter chose the provider for each call. The table carries its date for that reason.
- **Re-tests are partial.** Only the cheapest hits were sent to fresh tasks. A dearer setting marked "never sent to fresh tasks" may well hold its bar.
- **Wrong keys cut both ways.** Table 8 lists keys we doubt. We did not correct any key, so every score counts them as wrong.

## Use it

```
pip install "pydantic>=2"
python -m router options --kind maths
python -m router route --kind science --bar 95 "Why does ice float on water?"
python -m router table --kind exam --top 10
python -m router report --only t3
python -m pytest -q
```

Nothing here calls a model or the network. To add a model, answer the same 400 tasks with it and add its row to `data/measured_capability.json`. The script that calls the models is not in this repository yet.
