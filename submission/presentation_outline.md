# Track-3 Presentation Outline — CityLearn Challenge 2023

**Team:** Erik Dyer, Ammaar Alam, Grace Sun
**Target length:** 3.5 minutes (Rubric §3, 15% of grade)
**Slide deck:** https://docs.google.com/presentation/d/1KWFq_GIIJjpZBKr2GlFCQfEB8K1NnWniI406G8iabVM/edit

Each slide is ~30 seconds. All five rubric questions (problem / importance / hardness / prior work + differentiator / approach+results+limitations) are covered.

---

## Slide 1 — Title + problem (30s)

**Title:** Generalizable RL for Neighborhood Battery Control — CityLearn Challenge 2023

**On-slide content:**
- Team: Erik Dyer, Ammaar Alam, Grace Sun
- Track: 3 (RL Competition)
- One schematic: 3-building microgrid with shared battery + solar PV

**Speaker notes (Rubric Q1: "What is the problem?"):**
> We control a neighborhood-scale battery storage system across three buildings with solar generation and stochastic electricity demand. At each hour, the agent outputs a continuous action in [−1, +1] per building — negative discharges, positive charges — to optimize a composite score across cost, carbon, peak demand, discomfort, and thermal resilience.

---

## Slide 2 — Why it matters (30s)

**On-slide content:**
- Buildings ≈ **28% of global CO₂** emissions (IEA)
- Distributed storage + PV = key grid-decarbonization lever
- CityLearn = realistic, standardized benchmark for multi-building coordination

**Speaker notes (Rubric Q2: "Why is it interesting and important?"):**
> Better battery coordination directly translates to lower peak loads, cheaper bills, and less carbon — real-world stakes. The CityLearn benchmark gives us a controlled testbed to study RL for a problem where rule-based and MPC methods currently dominate.

---

## Slide 3 — Why it's hard (30s)

**On-slide content (bullets):**
- Non-stationary demand + weather
- Partial observability (no perfect load forecast)
- Multi-building coordination, long horizons
- Reward specification is not given — we designed 3 variants
- **Naive RBC = 1.02 on `public_dev`** (baseline to beat)

**Speaker notes (Rubric Q3: "Why is it hard? Why do naive approaches fail?"):**
> Rule-based control does a reasonable job because the domain has exploitable structure, but it cannot adapt to unusual conditions. RL promises better generalization, but training is unstable across seeds, reward misspecification dominates early results, and the long-horizon credit assignment is brutal. A naive single-seed PPO run looks better than RBC but regresses on held-out splits.

---

## Slide 4 — Prior work + our differentiator (30s)

**On-slide content:**
- **CityLearn 2023 winner: CHESCA, 0.562 public / 0.565 private** (hierarchical optimization, not RL)
- Most leaderboard entries: rule-based or MPC variants
- **Our contribution**: a systematic ablation, not a new algorithm
  - 2 algorithms × 2 control topologies × 3 reward variants
  - Multi-seed runs (n=3–10) with 95% CI error bars
  - Cross-split generalization: `public_dev` → `phase_2` → `phase_3` (6-building held-out)

**Speaker notes (Rubric Q4: "Why hasn't it been solved? How does mine differ?"):**
> Track 3 has already been "solved" if the metric is leaderboard rank — CHESCA won with hierarchical optimization. Nobody has done a clean comparison of RL architectures and rewards with seeded error bars, though. That's the gap we fill. We don't claim to beat CHESCA; we claim to map which design choices matter for RL practitioners who come next.

---

## Slide 5 — Approach: architecture + algorithms (30s)

**On-slide content (schematic + labels):**
- **Centralized:** one controller observes all 3 buildings, emits 3 joint actions
- **Shared-DTDE:** one policy network, per-building observation windows, decentralized rollout with shared parameters
- Both **PPO** (on-policy, clipped objective) and **SAC** (off-policy, entropy-regularized)

**Speaker notes:**
> We implemented PPO from scratch on a shared policy net with per-building embeddings for the DTDE variants, and used Stable-Baselines-3 for centralized PPO. SAC is custom for both topologies. The centralized architecture gets full joint-action credit assignment; the DTDE version is the only one that transfers to the 6-building phase-3 held-out split — centralized policies are fundamentally non-portable there.

---

## Slide 6 — Approach: reward design (30s)

**On-slide content (compact table):**

| Variant | Cost weight | Carbon weight | Discomfort | Shaping |
|---|---|---|---|---|
| baseline | equal | equal | equal | raw composite |
| **reward_v1** | × 2 | × 1 | — | Cost-forward |
| **reward_v2** | × 1.5 | × 1.5 | × 1.5 | **Balanced + generalization-friendly** |

**Speaker notes:**
> Reward shaping had more impact than any single hyperparameter tweak. reward_v1 is tuned for public_dev; reward_v2 sacrifices ~1% on local but generalizes better to released splits.

---

## Slide 7 — Results + limitations (30s)

**On-slide content — THE figure from P4**: `method_comparison.png` (or `method_vs_chesca.png`) showing 7 methods on public_dev with CHESCA reference line.

**Headline numbers (from [local_main_results.csv](submission/results/local_main_results.csv)):**
- Best local: **SAC centralized reward_v1 = 0.527** (−48.4% vs RBC)
- CHESCA public reference: **0.562**
- Best on released phase-2: **SAC central reward_v2 = 0.653** (generalization gap of ~0.12)
- Best portable (phase-3): **SAC shared-DTDE = 0.774**

**Speaker notes (Rubric Q5: "Key components + results + limitations"):**
> Three takeaways. First: SAC beats PPO by ~30% on every matched split — off-policy sample efficiency matters on this horizon. Second: centralized beats shared-DTDE on public_dev, but only DTDE transfers to the 6-building held-out phase-3 split — architecture is a generalization bet. Third: we're below CHESCA's public number on our tuning split, but the cross-split gap means we do not claim an apples-to-apples leaderboard win. The honest contribution is the ablation, not a new SOTA.
>
> Limitations: single public_dev tuning split used to select final reward variants (risk of overfitting); compute-bound so we capped PPO-DTDE at 10 seeds and SAC-DTDE at 3.

---

## Slide-count + timing check

- 7 slides × 30s = **3:30** — at 3.5-min budget
- Every rubric question answered on at least one slide
- Figure anchors the results slide (Rubric §2 "one figure" deliverable also lives in this figure)

## Sources for numbers on slides

All results quoted in the speaker notes trace back to tracked CSVs under [submission/results/](submission/results/) — specifically:
- Method means + CI: [local_main_results.csv](submission/results/local_main_results.csv) and [method_comparison.csv](submission/results/method_comparison.csv)
- Cross-split: [cross_split_scores.csv](submission/results/cross_split_scores.csv) and [released_eval_main_results.csv](submission/results/released_eval_main_results.csv)
- CHESCA + Official RBC: [official_benchmark_reference.csv](submission/results/official_benchmark_reference.csv)
