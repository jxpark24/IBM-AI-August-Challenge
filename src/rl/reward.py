"""
Reward function (PDF section 11.4 - Sudeepa owns this file).

Jiwoo owns the CALL SIGNATURE; Sudeepa owns the numbers. Swapping the weights
must never require touching routing_env.py.

The reward is shaped to match the PRIMARY evaluation metric of PDF section 10.4,
priority-weighted timely delivery:

    sum(priority of bundles delivered before deadline) / sum(priority generated)

Delivering a 0.96-priority transient on time is worth roughly twenty times
delivering a 0.05-priority housekeeping file on time, and that ratio lives in the
reward rather than in a hand-written rule. This is the whole reason the project
uses RL instead of another shortest-path variant: the objective is mission value,
not arrival time.

Deliberate choice: late delivery still scores POSITIVE, just small. Science data
that arrives after its deadline is usually degraded, not worthless, and a reward
of exactly zero would make "drop it" and "deliver it late" indistinguishable.
"""

from dataclasses import dataclass
from typing import Optional

from src.models.bundle import DataBundle

# Terminal episode statuses. DELIVERED/EXPIRED are the bundle's own vocabulary
# (PDF section 7.4); NO_ROUTE is an environment outcome, not a bundle state.
DELIVERED = "DELIVERED"
EXPIRED = "EXPIRED"
DROPPED = "DROPPED"
NO_ROUTE = "NO_ROUTE"
TRUNCATED = "TRUNCATED"


@dataclass
class RewardConfig:
    """PROTOTYPE ASSUMPTIONS (PDF section 15). Freeze alongside the checkpoint."""

    deliver_on_time: float = 1.0    # x priority
    early_bonus: float = 0.5        # x priority x fraction of deadline unused
    deliver_late: float = 0.1       # x priority
    expire: float = -1.0            # x priority
    no_route: float = -1.0          # x priority
    step_cost: float = -0.01        # mild pressure against pointless hops
    invalid_action: float = -0.5    # masked away, so this should stay at zero
    retry: float = -0.05            # a link that failed cost real time
    energy_weight: float = 0.0      # off by default; PDF 10.4 "optional but valuable"


def hop_reward(candidate, bundle: DataBundle,
               config: Optional[RewardConfig] = None) -> float:
    """Reward for taking one hop that has not ended the episode."""
    config = config or RewardConfig()
    r = config.step_cost
    if config.energy_weight:
        r -= config.energy_weight * candidate.contact.energy_cost
    return r


def retry_reward(config: Optional[RewardConfig] = None) -> float:
    config = config or RewardConfig()
    return config.retry


def invalid_action_reward(config: Optional[RewardConfig] = None) -> float:
    config = config or RewardConfig()
    return config.invalid_action


def terminal_reward(bundle: DataBundle, status: str, now_s: float,
                    config: Optional[RewardConfig] = None) -> float:
    """Reward for however the episode ended. Everything scales with priority."""
    config = config or RewardConfig()
    p = bundle.science_priority

    if status == DELIVERED:
        if bundle.deadline_s is None or now_s <= bundle.deadline_s:
            budget = (bundle.deadline_s - bundle.created_s
                      if bundle.deadline_s is not None else 0.0)
            slack = ((bundle.deadline_s - now_s) / budget) if budget > 0 else 0.0
            slack = max(0.0, min(1.0, slack))
            # Early delivery is worth more than just-in-time: real missions want
            # the alert, not a photo finish against the deadline.
            return p * (config.deliver_on_time + config.early_bonus * slack)
        return p * config.deliver_late

    if status == EXPIRED:
        return p * config.expire
    if status in (NO_ROUTE, DROPPED):
        return p * config.no_route
    return 0.0  # TRUNCATED: scenario ran out, not the policy's fault
