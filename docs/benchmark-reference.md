# benchmark reference

This note records the official 2023 challenge reference numbers we care about when reading local results from this repo.

## scoring rule

- lower is better
- public and private scores come from different evaluation splits
- public leaderboard numbers were visible during the competition
- private leaderboard numbers determined the final winner

## official 2023 leaderboard reference

Source: Table 1 in the CityLearn 2023 winning paper, "Winning the 2023 CityLearn Challenge: A Community-Based Hierarchical Energy Systems Coordination Algorithm"

Paper URL:
- https://publications.polymtl.ca/59790/1/59790_Winning_2023.pdf

| method | private score | public score | private - public |
| --- | ---: | ---: | ---: |
| RBC baseline | 1.124 | 1.085 | +0.039 |
| CHESCA winner | 0.565 | 0.562 | +0.003 |
| Team 2 | 0.575 | 0.464 | +0.111 |
| Team 3 | 0.582 | 0.508 | +0.074 |
| CHESCA* post-deadline improvement | 0.548 | 0.522 | +0.026 |

## winner vs baseline

Using the official leaderboard values:

- private improvement from RBC to CHESCA: `1.124 - 0.565 = 0.559`
- public improvement from RBC to CHESCA: `1.085 - 0.562 = 0.523`
- private percentage reduction: `0.559 / 1.124 = 49.7%`
- public percentage reduction: `0.523 / 1.085 = 48.2%`

The useful headline is that the official winner cut the baseline score by about half on both the public and private evaluations.

## how to interpret local repo runs

The repo default dataset is `citylearn_challenge_2023_phase_2_local_evaluation`, which is an official 2023 local-evaluation release used for local validation and development. It is not the same thing as the hidden competition private leaderboard split.

Because of that:

- a local run from this repo is useful as a reproducible development baseline
- a local run should not be treated as directly comparable to the official public or private leaderboard values
- differences in dataset split, release packaging, and evaluation scope can move the score even when the controller class is the same

During local validation of this repo foundation, the built-in `BasicRBC` baseline produced an average score of about `1.0226` on the default local-evaluation dataset. That number is a local benchmark sanity check, not an official leaderboard number.

## practical takeaway for this repo

When looking at outputs from `make train-rbc`:

- compare runs in this repo against each other first
- use the official 2023 leaderboard numbers only as an external reference point
- keep the public/private distinction explicit whenever a result is discussed in slides, notes, or the final report
