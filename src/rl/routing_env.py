"""
Next-hop routing environment (PDF section 9.2 / Appendix B4).

    action       Discrete(14) - the id of the node to send the bundle to next
    mask         action_masks() -> 14 booleans, physics-feasible hops only
    observation  172 floats, layout frozen in observation.py
    episode      one bundle from creation to delivery, expiry or scenario end
    fallback     an invalid or unavailable choice calls the temporal baseline
                 rather than dropping the bundle

WHY AN ACTION IS A NODE, NOT A CONTACT
A contact-indexed action space would change size every step and no fixed policy
network could consume it. Fixing the action to the 14 node ids keeps the head a
constant shape; masking supplies the "which of these exist right now" part. This
is the "fixed IDs + action masks" mitigation for dynamic-action instability in
PDF section 15.

WHERE WAITING WENT
There is no explicit wait action, and that is intentional. Choosing a node whose
contact opens later IS waiting - the clock advances to the window, the bundle
sits in storage, and the hop executes. Store-and-forward is expressed through
the choice of hop, exactly as it is in the temporal baseline, so the agent and
the benchmark are playing the same game (PDF section 8.4).

Gymnasium is an OPTIONAL import. The environment is pure-stdlib underneath so
tests and the demo run on a clean machine with nothing installed (PDF section 15:
"bundle orbital parameters, model checkpoint and saved scenarios; run offline").
Install gymnasium and it becomes a real gymnasium.Env that MaskablePPO accepts.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from src.models.bundle import DataBundle
from src.models.node import GROUND_IDS, NUM_NODES, NODE_NAMES, SCIENCE_ID
from src.rl.candidates import CandidateLink, enumerate_candidates
from src.rl.observation import OBS_SIZE, Normaliser, build_observation
from src.rl.reward import (
    DELIVERED, DROPPED, EXPIRED, NO_ROUTE, TRUNCATED,
    RewardConfig, hop_reward, invalid_action_reward, retry_reward,
    terminal_reward,
)
from src.rl.scenario import ScenarioConfig, make_scenario
from src.routing.temporal_baseline import earliest_arrival

try:  # optional
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover - depends on the machine, not the logic
    gym = None
    spaces = None
    _HAS_GYM = False

try:  # optional
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None
    _HAS_NUMPY = False

_EnvBase = gym.Env if _HAS_GYM else object


class RoutingEnv(_EnvBase):
    """One bundle, one dynamic contact plan, one next-hop decision at a time."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        stage: Optional[int] = None,
        seed: Optional[int] = None,
        normaliser: Optional[Normaliser] = None,
        reward_config: Optional[RewardConfig] = None,
        max_hops: int = 8,
        scenario_factory=make_scenario,
        max_reset_attempts: int = 20,
    ):
        self._config = config
        self._stage = stage
        self._normaliser = normaliser or Normaliser()
        self._reward_config = reward_config or RewardConfig()
        self._max_hops = int(max_hops)
        self._scenario_factory = scenario_factory
        self._max_reset_attempts = int(max_reset_attempts)

        self._rng = random.Random(seed)

        if _HAS_GYM:
            self.action_space = spaces.Discrete(NUM_NODES)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self._scn = None
        self._bundle: Optional[DataBundle] = None
        self._now_s = 0.0
        self._holder = SCIENCE_ID
        self._candidates: Dict[int, CandidateLink] = {}
        self._episode = _EpisodeStats()

    # --- gymnasium API ------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Draw a fresh scenario. Same seed => same episode, always (PDF 10.5)."""
        if seed is not None:
            self._rng = random.Random(seed)

        # A scenario where the bundle cannot move even once teaches nothing and
        # produces an all-False mask, which crashes MaskablePPO. Redraw instead.
        for _ in range(self._max_reset_attempts):
            self._start_episode()
            if self._candidates:
                break
        else:
            raise RuntimeError(
                "no routable scenario in %d attempts - the scenario "
                "configuration is probably infeasible" % self._max_reset_attempts)

        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        t0 = time.perf_counter()
        self._episode.decisions += 1

        reward = 0.0
        cand = self._candidates.get(action)

        # --- fallback (PDF section 9.2: never drop on an invalid choice) -----
        if cand is None:
            self._episode.invalid_actions += 1
            reward += invalid_action_reward(self._reward_config)
            fallback = self.baseline_action()
            if fallback is None:
                # Nothing was reachable at all; end honestly rather than stall.
                self._episode.decision_s += time.perf_counter() - t0
                return self._finish(NO_ROUTE, reward)
            self._episode.fallbacks += 1
            cand = self._candidates[fallback]

        # --- execute the hop -------------------------------------------------
        outcome = self._execute(cand)
        self._episode.decision_s += time.perf_counter() - t0

        if outcome == "RETRY":
            reward += retry_reward(self._reward_config)
            self._refresh_candidates()
            terminal = self._check_terminal()
            if terminal:
                return self._finish(terminal, reward)
            return self._obs(), reward, False, False, self._info()

        reward += hop_reward(cand, self._bundle, self._reward_config)
        self._refresh_candidates()

        terminal = self._check_terminal()
        if terminal:
            return self._finish(terminal, reward)

        return self._obs(), reward, False, False, self._info()

    def action_masks(self) -> List[bool]:
        """sb3-contrib MaskablePPO calls this by name. Length is always 14."""
        return [i in self._candidates for i in range(NUM_NODES)]

    def render(self):
        b = self._bundle
        path = " -> ".join(NODE_NAMES[i] for i in b.route_history)
        return "t=%8.1fs  %s  %s  pri %.2f  %.0f MB  deadline %.0f  [%s]" % (
            self._now_s, b.bundle_id, path, b.science_priority,
            b.remaining_bytes / 1e6, b.deadline_s or -1, b.status)

    def close(self):
        pass

    # --- the deterministic comparator, also used as the fallback -------------

    def baseline_action(self) -> Optional[int]:
        """Next hop the temporal earliest-arrival router would take (PDF 9.1).

        Two jobs in one method, on purpose:
          1. the safety net when the policy picks something unusable
          2. the benchmark policy in the baseline-vs-AI experiment

        They must be the same code. If the fallback were a simplified version,
        "RL + fallback" would be measuring a policy nobody benchmarked.
        """
        if self._bundle is None or not self._candidates:
            return None

        route = earliest_arrival(
            self._scn.plan, self._holder, self._scn.ground_ids,
            self._bundle.remaining_bytes, start_s=self._now_s,
            deadline_s=self._bundle.deadline_s,
        )
        if route is not None:
            nxt = route.next_hop()
            if nxt in self._candidates:
                return nxt

        # No deadline-feasible route. Still move the data - late science beats
        # lost science - so fall back to whichever legal hop lands earliest.
        route = earliest_arrival(
            self._scn.plan, self._holder, self._scn.ground_ids,
            self._bundle.remaining_bytes, start_s=self._now_s, deadline_s=None,
        )
        if route is not None:
            nxt = route.next_hop()
            if nxt in self._candidates:
                return nxt

        return min(self._candidates.values(), key=lambda c: c.arrival_s).dest_id

    # --- internals ----------------------------------------------------------

    def _start_episode(self):
        scenario_seed = self._rng.randrange(2 ** 31)
        self._scn = self._scenario_factory(
            seed=scenario_seed, config=self._config, stage=self._stage)
        self._bundle = self._scn.bundles[0]
        self._now_s = self._bundle.created_s
        self._holder = self._bundle.source_id
        self._episode = _EpisodeStats(scenario_seed=scenario_seed)
        self._refresh_candidates()

    def _refresh_candidates(self):
        self._candidates = enumerate_candidates(
            self._scn.plan, self._scn.nodes, self._holder, self._now_s,
            self._bundle.remaining_bytes, self._scn.horizon_s)

    def _execute(self, cand: CandidateLink) -> str:
        """Move the bundle, or fail the link and lose the time trying."""
        contact = cand.contact

        # Link reliability (weather, pointing, health). A failed attempt still
        # consumes the window - that is what makes retries costly.
        if contact.reliability < 1.0 and self._rng.random() > contact.reliability:
            self._bundle.retries += 1
            self._now_s = cand.depart_s + cand.tx_s
            self._episode.retries += 1
            return "RETRY"

        size = self._bundle.remaining_bytes
        contact.residual_capacity_bytes -= size

        src_node = self._scn.nodes[self._holder]
        src_node.storage_used_bytes = max(0, src_node.storage_used_bytes - size)

        self._episode.wait_s += cand.wait_s
        self._episode.energy += contact.energy_cost

        self._now_s = cand.arrival_s
        self._holder = cand.dest_id
        self._bundle.current_holder = cand.dest_id
        self._bundle.route_history.append(cand.dest_id)
        self._episode.hops += 1

        dst_node = self._scn.nodes[cand.dest_id]
        if not dst_node.is_ground:
            dst_node.storage_used_bytes = min(
                dst_node.storage_capacity_bytes,
                dst_node.storage_used_bytes + size)
            self._bundle.status = "STORED"
        return "OK"

    def _check_terminal(self) -> Optional[str]:
        if self._holder in self._scn.ground_ids:
            return DELIVERED
        if self._bundle.is_expired(self._now_s):
            return EXPIRED
        if self._now_s >= self._scn.horizon_s:
            return TRUNCATED
        if self._episode.hops >= self._max_hops:
            return TRUNCATED
        if not self._candidates:
            return NO_ROUTE
        return None

    def _finish(self, status: str, reward: float):
        self._bundle.status = DELIVERED if status == DELIVERED else (
            EXPIRED if status == EXPIRED else DROPPED)
        reward += terminal_reward(
            self._bundle, status, self._now_s, self._reward_config)
        self._episode.status = status
        truncated = status == TRUNCATED
        return self._obs(), reward, not truncated, truncated, self._info()

    def _obs(self):
        obs = build_observation(
            self._bundle, self._holder, self._now_s, self._scn.nodes,
            self._scn.plan, self._candidates, self._normaliser)
        if _HAS_NUMPY:
            return np.asarray(obs, dtype=np.float32)
        return obs

    def _info(self) -> Dict:
        """Everything the experiment harness needs for PDF section 10.4 metrics,
        emitted every step so a crashed run still leaves usable rows."""
        b = self._bundle
        latency = self._now_s - b.created_s
        on_time = (self._episode.status == DELIVERED
                   and (b.deadline_s is None or self._now_s <= b.deadline_s))
        return {
            "bundle_id": b.bundle_id,
            "data_type": b.data_type,
            "science_priority": b.science_priority,
            "size_bytes": b.size_bytes,
            "deadline_s": b.deadline_s,
            "status": self._episode.status,
            "delivered": self._episode.status == DELIVERED,
            "on_time": on_time,
            "latency_s": latency,
            "hops": self._episode.hops,
            "retries": self._episode.retries,
            "waited_s": self._episode.wait_s,
            "energy": self._episode.energy,
            "invalid_actions": self._episode.invalid_actions,
            "fallbacks": self._episode.fallbacks,
            "decisions": self._episode.decisions,
            "decision_ms": self._episode.decision_s * 1000.0,
            "path": list(b.route_history),
            "now_s": self._now_s,
            "scenario_seed": self._episode.scenario_seed,
            "action_mask": self.action_masks(),
        }


class _EpisodeStats:
    __slots__ = ("hops", "retries", "wait_s", "energy", "invalid_actions",
                 "fallbacks", "decisions", "decision_s", "status",
                 "scenario_seed")

    def __init__(self, scenario_seed=None):
        self.hops = 0
        self.retries = 0
        self.wait_s = 0.0
        self.energy = 0.0
        self.invalid_actions = 0
        self.fallbacks = 0
        self.decisions = 0
        self.decision_s = 0.0
        self.status = "IN_FLIGHT"
        self.scenario_seed = scenario_seed
