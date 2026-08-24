# RL routing environment

*The mandatory AI contribution (PDF §9.2). A satellite has to decide, right now,
which neighbour to hand a file of science data to — and the right answer depends
on windows that have not opened yet.*

**Scope.** `src/rl/` only — one workstream (Jinwoo, PDF §11.3) inside the
five-person project. Training, reward tuning and evaluation are Sudeepa's
(PDF §11.4); orbital physics, the clock and the dashboard belong to Harshul and
Serafin. This document is the **interface contract** between those workstreams.

August Challenge 2026 · prototype deadline 31 August 2026

---

## 1. What the agent is actually deciding

Not "what is the shortest path". The graph does not sit still long enough for
that question to mean anything.

At a given moment the bundle sits on some node. A handful of neighbours are
reachable — some over a link that is open now, some over a link that opens in
four minutes. The agent picks one. That is the whole action.

The interesting part is that **waiting is a move**. From a real scenario:

```
next hop | legal | wait (s) | transmit (s) | arrives at | link
LEO1     |   yes |    153.1 |         11.2 |     1760.2 | OPTICAL_ISL
GEO1     |   yes |      0.0 |        280.5 |     1876.5 | GEO_RELAY
```

GEO1 is available immediately. LEO1 is not available for another two and a half
minutes. Choosing LEO1 — sitting in storage, doing nothing, and *then*
transmitting — still delivers the data **116 seconds earlier**.

A router that asks "who can I see right now?" cannot represent that move at all.
This is the store-and-forward idea from CCSDS Schedule-Aware Bundle Routing that
the whole project is built on (PDF §8.4).

---

## 2. The frozen interface

Changing anything in this section invalidates every trained checkpoint. It was
frozen on 20 August (PDF §12.2) and should not move.

| | |
|---|---|
| **Action** | `Discrete(14)` — the id of the node to send the bundle to next |
| **Mask** | `env.action_masks()` → 14 booleans |
| **Observation** | 172 floats, all in `[0, 1]` |
| **Episode** | one bundle, ends on delivery / expiry / no route / horizon |
| **Reward** | scales with `science_priority`; see `reward.py` |

Node ids are fixed and shared with the router and the physics config:

```
0       science satellite      SCI
1-8     LEO relays             LEO1..LEO8
9-10    GEO relays             GEO1, GEO2
11-13   ground stations        GNDA, GNDB, GNDC
```

### Observation layout

```
[0:4]     bundle:    priority, size, deadline remaining, age
[4:18]    one-hot of the node currently holding the bundle
[18:172]  14 blocks x 11 features, indexed by NODE ID
```

Block *j* always describes node *j*, reachable or not. An unreachable node is
all zeros beginning with `valid=0`. A "top-5 reachable neighbours" layout would
be smaller, but the meaning of each slot would shift between steps and the
policy could never learn *"GEO2 is slow"*.

The 11 per-candidate features are listed in
`observation.CANDIDATE_FEATURE_NAMES`, so nothing has to count array offsets.

---

## 3. Why an action is a node, not a contact

Contacts appear and disappear constantly. A contact-indexed action space would
change size every step, and no fixed policy network can consume that.

Fixing the action to the 14 node ids keeps the output head a constant shape;
masking supplies the *"which of these exist right now"* part. This is the
`fixed IDs + action masks` mitigation for dynamic-action instability in PDF §15,
and it is why `test_every_masked_valid_action_is_actually_executable` exists.

### Masking encodes physics, never policy

| masked out | left legal |
|---|---|
| the window is shut | this route will miss the deadline |
| the transfer cannot finish in time | this relay looks congested |
| the relay is dead | this gateway is under cloud |
| the receiver has no storage | this path costs more energy |

Everything on the right the agent has to **learn**. And a mask driven by the
deadline would go all-False the moment a bundle became doomed — which crashes
`MaskablePPO` instead of teaching it anything.

---

## 4. Files

| file | owner | what it is |
|---|---|---|
| `candidates.py` | Jinwoo | what counts as a legal next hop — **one** definition, shared by mask, observation and `step()` |
| `routing_env.py` | Jinwoo | the environment, masking, baseline fallback |
| `harness.py` | Jinwoo | baseline-vs-AI comparison over identical seeds |
| `observation.py` | **Sudeepa** | feature normalisation — shape and order are frozen, the numbers are not |
| `reward.py` | **Sudeepa** | reward weights — call signature is frozen |
| `scenario.py` | *temporary* | mock dynamic graph; **deleted** when the real contact generator lands |

`candidates.py` exists to kill one specific bug: if the mask, the observation and
`step()` each decide independently what is legal, they will drift, and the
symptom is the worst kind — the agent picks an action the mask allowed, `step()`
refuses it, and the training curve quietly degrades with no error anywhere.

---

## 5. Plugging in the agent

```python
from sb3_contrib import MaskablePPO
from src.rl.harness import HOLDOUT_SEEDS, baseline_policy, compare, render_table
from src.rl.routing_env import RoutingEnv

model = MaskablePPO("MlpPolicy", RoutingEnv(stage=2), verbose=1)
model.learn(200_000)

def rl_policy(env):
    return int(model.predict(env._obs(), action_masks=env.action_masks(),
                             deterministic=True)[0])

print(render_table(compare({"baseline": baseline_policy, "rl": rl_policy},
                           seeds=HOLDOUT_SEEDS, stage=5)))
```

`gymnasium` is an **optional** import. Without it the environment is pure
stdlib and the tests and demo still run — the demo machine may have nothing
installed (PDF §15: *"run offline"*). With it, `RoutingEnv` is a real
`gymnasium.Env` that `MaskablePPO` accepts.

### Curriculum (PDF §9.4)

`RoutingEnv(stage=N)`, difficulty added one source at a time:

| stage | adds |
|---|---|
| 1 | static graph, identical bundles — obvious optimal paths |
| 2 | time-varying contacts |
| 3 | congestion |
| 4 | science priorities and deadlines |
| 5 | optical weather and relay failures |

Held-out evaluation uses `HOLDOUT_SEEDS` (1000–1099), disjoint from
`TRAIN_SEEDS` (0–399) by construction.

---

## 6. Known problem: the baseline has no headroom

On held-out seeds the temporal baseline scores:

| stage | priority-weighted timely delivery |
|---|---|
| 1–4 | **1.000** |
| 5 | 0.982 |

There is essentially nothing for the RL agent to win. This is PDF §15's
*"AI fails to beat a strong baseline"* risk, arriving at the scenario-design
level rather than the agent level.

**Cause: one bundle per episode.** With nothing else competing for contact
capacity, earliest-arrival *is* optimal — there is no tension for a smarter
policy to exploit. Every §10.3 scenario where AI is supposed to win (congestion,
priority burst, combined stress) requires **several bundles contending for the
same finite windows**.

`ScenarioConfig.bundles_per_episode` already exists; the environment currently
ignores it. That is the next task, and it is on the critical path for the
27 August *"AI vs baseline held-out"* milestone — not for the 24 August gate,
which is about action validity and is met.

---

## 7. Running it

This machine has no bare `python` — macOS ships `python3` only.

```
python3 demo_rl_env.py              # decision walkthrough + gate check
```

The demo needs nothing installed. The tests need `pytest`, which lives in the
project `.venv` (Python 3.9.6, at the main checkout — **not** inside a git
worktree):

```
~/myProject/IBM-AI-August-Challenge/.venv/bin/python -m pytest tests/ -q
```

31 tests, ~0.5 s, no third-party dependencies.
