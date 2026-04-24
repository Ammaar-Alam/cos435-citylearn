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

| Variant | What it is | Why |
|---|---|---|
| `reward_v0` (baseline) | CityLearn's built-in default (negative net-consumption proxy) | Sanity baseline; the training signal does not match the scoring rule |
| **`reward_v1`** | Hand-crafted 8-term weighted penalty mirroring the **official 2023 scoring weights**: 0.30·comfort + 0.15·outage_comfort + 0.15·outage_unserved + 0.10·carbon + 0.075·{ramping, load_factor, daily_peak, all_time_peak} | Train on the same thing the competition scores on |
| **`reward_v2`** | `reward_v1` + 0.05·\|ΔSoC\| + 0.05·1[sign_flip] | Battery smoothness penalties — discourages bang-bang charge/discharge; empirically improves cross-split generalization |

**Speaker notes:**
> Instead of inventing weights, reward_v1 literally mirrors the 8-term weighted penalty the competition uses to score submissions, so the agent is trained directly on the objective it's evaluated on. reward_v2 adds two battery-behavior terms that penalize SoC swings and charge-discharge sign flips between consecutive steps. Without them, SAC learns bang-bang policies that are locally optimal on public_dev but brittle on held-out splits; with them, the policy is steadier and generalizes better. The reward-ablation axis is run only on SAC-central — we didn't have compute to matched-ablate PPO.

---

## Slide 7 — Results + limitations (30s)

**On-slide content — THE figure (Rubric §2 deliverable)**: [`submission/figures/per_split_scores.png`](submission/figures/per_split_scores.png) — the single strongest figure we have, because it shows **all methods on both held-out tiers in one panel**: 3 `phase_2_online_eval` splits (same-size generalization, 3 buildings) on the left, 3 `phase_3` splits (cross-size generalization, 6 buildings) on the right, with a visual separator and a "centralized models not portable" annotation in the phase_3 region. The dashed red line is the CHESCA 2023 public-leaderboard reference (0.562). Backup / supporting figures available: [`cross_split_comparison.png`](submission/figures/cross_split_comparison.png) (phase_2 bar chart), [`generalization_gap.png`](submission/figures/generalization_gap.png) (public_dev → phase_2 paired bars).

**Headline numbers (from [released_eval_main_results.csv](submission/results/released_eval_main_results.csv)):**

On `phase_2_online_eval` (3 buildings, held-out):
- RBC baseline: 1.087
- PPO central: 0.873
- **SAC central reward_v2: 0.653 (best)**
- SAC DTDE: 0.677
- CHESCA public reference (different eval): 0.562

On `phase_3` (6 buildings, held-out — only portable methods):
- RBC baseline: 1.114
- PPO DTDE: 0.843
- **SAC DTDE: 0.774 (best)**

**Speaker notes (Rubric Q5: "Key components + results + limitations"):**
> The headline panel is the held-out `phase_2_online_eval` split — same 3-building setup as training but weather and demand the agent has never seen. RBC sits at 1.09, PPO central at 0.87, SAC variants cluster at 0.65-0.68. The dashed line at 0.562 is CHESCA's 2023 public-leaderboard score — benchmark context, not a head-to-head number, because CHESCA was scored on the original competition server we cannot re-run.
>
> The harder test is `phase_3` — held-out AND a different 6-building cluster, not the 3 we trained on. Only the shared-DTDE variants can execute here: a centralized policy's input and output layers are wired for exactly 3 buildings, so feeding it a 6-building observation is a shape mismatch, not a degradation. SAC-DTDE generalizes best at 0.774, PPO-DTDE at 0.843. This is our cleanest architectural claim: **if deployment might see a different building count than training, choose shared-DTDE**.
>
> Limitations: single `public_dev` tuning split was used to select reward variants (overfitting risk); compute-bound so we capped PPO-DTDE and SAC-DTDE at 10 and 3 seeds respectively; reward-variant ablation was run only on SAC-central, not on PPO-central. CHESCA's line is reference, not a competitive bar we're claiming to have cleared.

---

## Figure caption (for PowerPoint / Slides paste — Rubric §2)

> **Figure.** CityLearn 2023 Track-3 `average_score` (↓ better) across six released held-out evaluation datasets: three `phase_2_online_eval` splits (left, same 3-building setup as training) and three `phase_3` splits (right, six-building cluster, shaded). Each line is one method: rule-based control (grey), PPO centralized (blue), three SAC centralized reward variants (orange/green/red), and SAC shared-DTDE (purple). The dashed red reference line at 0.562 is CHESCA's 2023 public-leaderboard score, shown as benchmark context (not same eval). **Takeaways**: (1) on `phase_2` (3 buildings), all SAC variants cluster around 0.65-0.68 and PPO-central around 0.87 — RL reduces the RBC score of ~1.09 by 30-40 % on held-out data. (2) On `phase_3` (6 buildings), only shared-DTDE policies can run; SAC-DTDE is the best portable variant at 0.774. Centralized policies are architecturally non-portable to a different building count, which is the headline architectural trade-off.

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
