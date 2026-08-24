"""
Baseline-vs-AI experiment harness (PDF sections 10.2, 10.4, 10.5).

Jiwoo owns this (PDF section 11.3). Its job is to make the comparison FAIR, and
fairness here is mostly one rule:

    every policy sees exactly the same scenarios, in the same order, from the
    same seeds (PDF section 10.5)

Because RoutingEnv derives its scenario from the seed alone, running two
policies over the same seed list gives them byte-identical traffic, contacts,
weather and failures. Any difference in the metrics is then attributable to the
policy - which is the entire claim the competition submission rests on.

Adding the RL agent is one function:

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load("checkpoints/ppo_v1")
    def rl_policy(env):
        return int(model.predict(env._obs(), action_masks=env.action_masks(),
                                 deterministic=True)[0])
    compare({"baseline": baseline_policy, "rl": rl_policy}, HOLDOUT_SEEDS)
"""

import random
import statistics
from typing import Callable, Dict, List, Optional, Sequence

from src.rl.routing_env import RoutingEnv

# PDF section 10.5: "keep a held-out group of scenarios not used in RL training".
# Disjoint by construction, declared here so nobody has to remember the split.
TRAIN_SEEDS = tuple(range(0, 400))
HOLDOUT_SEEDS = tuple(range(1000, 1100))

URGENT_PRIORITY = 0.8  # the "high-priority" class for urgent p95 latency


# --- policies (PDF section 10.2) --------------------------------------------

def random_unmasked_policy(env):
    """Picks any node id at all. NOT a serious policy - it is the control that
    shows what action masking is worth. Its valid-action rate is the number the
    24 August gate is really about."""
    return random.randrange(len(env.action_masks()))


def random_masked_policy(env):
    """The sanity floor of PDF section 10.2: a valid random next hop. The AI must
    comfortably beat this or the AI is decorative."""
    valid = [i for i, ok in enumerate(env.action_masks()) if ok]
    return random.choice(valid) if valid else 0


def baseline_policy(env):
    """Temporal earliest-arrival / CGR-like. The primary non-AI benchmark."""
    action = env.baseline_action()
    return action if action is not None else 0


BUILTIN_POLICIES = {
    "random_unmasked": random_unmasked_policy,
    "random_masked": random_masked_policy,
    "baseline": baseline_policy,
}


# --- running ----------------------------------------------------------------

def run_episode(env: RoutingEnv, policy: Callable, seed: int) -> Dict:
    """One bundle, one policy. Returns the tidy row PDF appendix B5 asks for."""
    env.reset(seed=seed)
    info = env._info()
    for _ in range(64):
        _, _, terminated, truncated, info = env.step(policy(env))
        if terminated or truncated:
            break
    row = dict(info)
    row["seed"] = seed
    return row


def evaluate(policy: Callable, seeds: Sequence[int] = HOLDOUT_SEEDS,
             stage: int = 5, rng_seed: int = 0, **env_kwargs) -> List[Dict]:
    """Run one policy over a fixed seed list. Stochastic policies get their own
    seeded RNG so even the random floor is reproducible."""
    random.seed(rng_seed)
    env = RoutingEnv(stage=stage, **env_kwargs)
    return [run_episode(env, policy, seed) for seed in seeds]


def _percentile(values: List[float], q: float) -> float:
    """p95 without numpy. Nearest-rank; on ~100 samples the interpolation
    method makes no difference worth a dependency."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def aggregate(rows: List[Dict]) -> Dict:
    """PDF section 10.4 metrics. Primary metrics first."""
    if not rows:
        return {}

    total_priority = sum(r["science_priority"] for r in rows)
    on_time_priority = sum(r["science_priority"] for r in rows if r["on_time"])

    delivered = [r for r in rows if r["delivered"]]
    urgent = [r for r in delivered
              if r["science_priority"] >= URGENT_PRIORITY]
    latencies = [r["latency_s"] for r in delivered]

    decisions = sum(r["decisions"] for r in rows)
    invalid = sum(r["invalid_actions"] for r in rows)

    return {
        # PRIMARY
        "priority_weighted_timely": (on_time_priority / total_priority
                                     if total_priority else 0.0),
        "urgent_p95_latency_s": _percentile(
            [r["latency_s"] for r in urgent], 0.95),
        "deadline_success": sum(1 for r in rows if r["on_time"]) / len(rows),
        # CORE
        "delivery_ratio": len(delivered) / len(rows),
        "mean_latency_s": statistics.fmean(latencies) if latencies else float("nan"),
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_hops": statistics.fmean([r["hops"] for r in rows]),
        "mean_retries": statistics.fmean([r["retries"] for r in rows]),
        # DIAGNOSTIC
        "valid_action_rate": (decisions - invalid) / decisions if decisions else 1.0,
        "fallbacks": sum(r["fallbacks"] for r in rows),
        "mean_decision_ms": statistics.fmean(
            [r["decision_ms"] / max(1, r["decisions"]) for r in rows]),
        "episodes": len(rows),
    }


def compare(policies: Optional[Dict[str, Callable]] = None,
            seeds: Sequence[int] = HOLDOUT_SEEDS,
            stage: int = 5, **env_kwargs) -> Dict[str, Dict]:
    """Run several policies over identical scenarios and aggregate each."""
    policies = policies or BUILTIN_POLICIES
    return {name: aggregate(evaluate(fn, seeds, stage, **env_kwargs))
            for name, fn in policies.items()}


# --- reporting ---------------------------------------------------------------

ROWS = [
    ("priority_weighted_timely", "Priority-weighted timely delivery", "{:.3f}", "PRIMARY"),
    ("urgent_p95_latency_s",     "Urgent p95 latency (s)",            "{:.1f}", "PRIMARY"),
    ("deadline_success",         "Deadline success rate",             "{:.3f}", "PRIMARY"),
    ("delivery_ratio",           "Delivery ratio",                    "{:.3f}", "core"),
    ("mean_latency_s",           "Mean latency (s)",                  "{:.1f}", "core"),
    ("p95_latency_s",            "p95 latency (s)",                   "{:.1f}", "core"),
    ("mean_hops",                "Mean hops",                         "{:.2f}", "core"),
    ("mean_retries",             "Mean reroutes/retries",             "{:.2f}", "diag"),
    ("valid_action_rate",        "Valid action rate",                 "{:.4f}", "GATE"),
    ("mean_decision_ms",         "Decision time (ms)",                "{:.3f}", "diag"),
]


def render_table(results: Dict[str, Dict], title: str = "") -> str:
    """Plain text, no plotting library - it has to work on the demo machine."""
    names = list(results)
    width = max(36, max((len(r[1]) for r in ROWS), default=20) + 2)
    col = 18

    out = []
    if title:
        out.append(title)
    out.append("=" * (width + col * len(names) + 10))
    out.append("metric".ljust(width) + "".join(n.rjust(col) for n in names)
               + "      priority")
    out.append("-" * (width + col * len(names) + 10))
    for key, label, fmt, tag in ROWS:
        line = label.ljust(width)
        for name in names:
            value = results[name].get(key)
            line += (fmt.format(value) if value is not None else "-").rjust(col)
        out.append(line + "      " + tag)
    out.append("=" * (width + col * len(names) + 10))
    out.append("episodes: %d   scenarios identical across policies (PDF 10.5)"
               % next(iter(results.values())).get("episodes", 0))
    return "\n".join(out)
