COS 435 / ECE 433 : Introduction to RL Due: March 20 , 2026

# Project Proposal: Generalizable RL for Neighborhood

# Battery Control in CityLearn

**Authors.** Erik Dyer (erikdyer@princeton.edu), Ammaar Alam (ah0952@princeton.edu),
Grace Sun (gs0625@princeton.edu)

**Project type:** 3. Competition-based project (The CityLearn Challenge 2023 ).

**Description** We plan to study CityLearn, an open-source gynasium environment where
agents control neighborhood batteries for homes with solar generation and varying electric-
ity demand. During each step, the controller outputs a continuous actionat∈[−1, 1]per
building, where - 1 is full discharge and + 1 is full charge, to improve neighborhood-level
metrics like electricity cost, carbon emissions, and load.

This challenge is preferred since it is a
clear sequential decision-making problem
coupled with delayed effects, such as how
charging now affects future choices, so the
value of an action depends on how it affects
later states and opportunities for control
[ 1 , 3 ].

We will use the CityLearn environment and
starter code as an initial framework [ 2 ]. Our
initial milestone leverage Stable-Baselines 3
to implement working baseline controllers,
such as CityLearn’s built-in Rule-Based
Controller (RBC) and RL agents like PPO
and SAC. Then, we will progress to testing select changes to see if they improve per-
formance and generalization (e.g., short-horizon forecast, decentralized and centralized
control, or modified reward design). Our primary research question is whether a relatively
simple RL pipeline can outperform classical hierarchical optimization (e.g., CHESCA,
2023 winner) and rule-based strategies while remaining robust on held-out conditions
(expanding the RL pipeline as needed for strech goals).

We anticipate the main challenges to be sample efficiency, reward misspecification, in-
stability across various seeds, multi-agent assignment in decentralized modes, and the
dominance of existing non-RL methods (primarily rule-based or model-predictive). As a
result, we hope to focus on reproducible experiments with clear baselines and evaluations
compared to large hyperparameters sweeps. Furthermore, RL is still attractive here since
it can learn control policies from direct interaction without an explicit system model,
meaning it can generalize across buildings, seasons, or forecasts. Our main goal is pro-
ducing a strong baseline with meaningful improvement through feature or reward design,
and a breakdown of where learned controllers succeed vs fail. Notably, this problem
matters since better coordination of distributed batteries and solar generations have direct
real-world implications in reducing electricity costs, lower carbon emissions, and easing
peak demand on the grid (electrifying the grid!).

The project will be subdivided among the members. Ammaar will work on environment
setup and SAC implementation, Erik on PPO implementation and hyperparameter tuning,
and Grace will manage the RBC baseline and evaluation pipeline. We will also follow
standard group project protocols like meeting biweekly, maintaing code in a shared repo,
and keeping a log to track design choices (note: workflow split-up is more subject to
change, with this serving as general idea but can adjust as needed).


Project Proposal: Generalizable RL for Neighborhood Battery Control in CityLearn

```
COS 435 / ECE 433 : Introduction to RL
Due: March 20 , 2026
```
## References

[ 1 ]CityLearn ( 2023 ). Citylearn challenge 2023 .https://www.citylearn.net/citylearn_challenge/
2023.html.

[ 2 ]Intelligent Environments Lab ( 2025 ). Citylearn github repository. https://github.com/
intelligent-environments-lab/CityLearn.

[ 3 ]Vázquez-Canteli, J. R., Khamis, J., Dey, S., Henze, G., and Nagy, Z. ( 2020 ). Citylearn: Standard-
izing research in multi-agent reinforcement learning for demand response and urban energy
management. arXiv preprint arXiv: _2012_. _10504_.

### 2


