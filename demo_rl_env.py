"""
Watch the routing environment work, and check the 24 August gate.

    ./.venv/bin/python demo_rl_env.py

No trained agent here - Sudeepa owns training (PDF section 11.4). What this
shows is that the environment the agent will train in is sound:

  1. the decision the agent actually faces, at one real moment
  2. action masking doing its job (the 24 August gate)
  3. the baseline-vs-random comparison the AI has to slot into
"""

from src.models.node import NODE_NAMES
from src.rl.harness import (
    BUILTIN_POLICIES, HOLDOUT_SEEDS, baseline_policy, compare, render_table,
)
from src.rl.routing_env import RoutingEnv

LINE = "=" * 100


# 1 --------------------------------------------------------------------------
print("\n" + LINE)
print("1.  THE DECISION, AT ONE REAL MOMENT")
print(LINE)
print("""
The bundle is on the science satellite. Several hops are legal. Some are open
right now, some need waiting. The naive move is "take a link that is open" -
compare the transmit times and arrival times before agreeing with it.
""")

env = RoutingEnv(stage=2, seed=4)
env.reset()
bundle = env._bundle
print("bundle %s   %s   %.0f MB   priority %.2f   deadline t=%.0f   now t=%.0f\n"
      % (bundle.bundle_id, bundle.data_type, bundle.size_bytes / 1e6,
         bundle.science_priority, bundle.deadline_s, env._now_s))

print("  next hop | legal | wait (s) | transmit (s) | arrives at | link")
print("  " + "-" * 74)
for node_id in range(len(env.action_masks())):
    cand = env._candidates.get(node_id)
    if cand is None:
        continue
    print("  %-8s |   yes | %8.1f | %12.1f | %10.1f | %s"
          % (NODE_NAMES[node_id], cand.wait_s, cand.tx_s, cand.arrival_s,
             cand.contact.link_type))

blocked = [NODE_NAMES[i] for i, ok in enumerate(env.action_masks()) if not ok]
print("\n  masked out (no feasible contact): %s"
      % (", ".join(blocked) if blocked else "none"))
# Derive the narration from the data rather than hardcoding numbers that go
# stale the moment a scenario constant changes.
_geo = min((c for c in env._candidates.values()
            if c.contact.link_type == "GEO_RELAY"), key=lambda c: c.arrival_s)
# The most patient LEO hop that still beats GEO: the strongest illustration that
# an open link is not the same as a good link.
_waiters = [c for c in env._candidates.values()
            if c.contact.link_type == "OPTICAL_ISL"
            and c.wait_s > 0 and c.arrival_s < _geo.arrival_s]
_leo = max(_waiters, key=lambda c: c.wait_s) if _waiters else min(
    (c for c in env._candidates.values()
     if c.contact.link_type == "OPTICAL_ISL"), key=lambda c: c.arrival_s)

print("""
GEO is open instantly, then spends %.0f s pushing the data and lands at t=%.0f.
%s is not open yet. Sitting in storage for %.0f s and then transmitting for %.0f s
still lands at t=%.0f -- %.0f s EARLIER than the link that was available all along.

An open link is not the same as a good link. The agent can express that only
because choosing a not-yet-open contact IS the wait (PDF section 8.4); a router
that asks "who can I see right now" cannot represent this move at all.
""" % (_geo.tx_s, _geo.arrival_s, NODE_NAMES[_leo.dest_id], _leo.wait_s,
       _leo.tx_s, _leo.arrival_s, _geo.arrival_s - _leo.arrival_s))


# 2 --------------------------------------------------------------------------
print(LINE)
print("2.  ONE EPISODE, BASELINE POLICY")
print(LINE + "\n")

# Seed 2 is an urgent transient that needs two LEO relays and 158 s of storage
# waiting -- the store-and-forward behaviour the whole project is about.
env = RoutingEnv(stage=5, seed=2)
env.reset()
print("  " + env.render())
while True:
    action = baseline_policy(env)
    _, reward, terminated, truncated, info = env.step(action)
    print("  -> %-5s reward %+.3f   t=%8.1f   %s"
          % (NODE_NAMES[info["path"][-1]], reward, info["now_s"], info["status"]))
    if terminated or truncated:
        break
print("\n  latency %.1f s   on time: %s   hops %d   waited %.0f s in storage"
      "   decision %.3f ms"
      % (info["latency_s"], info["on_time"], info["hops"], info["waited_s"],
         info["decision_ms"] / max(1, info["decisions"])))


# 3 --------------------------------------------------------------------------
print("\n" + LINE)
print("3.  GATE CHECK  -  24 AUGUST: 'agent selects valid masked actions'")
print(LINE)

results = compare(seeds=HOLDOUT_SEEDS, stage=5)
print("\n" + render_table(results, "Stage 5: congestion + optical weather + relay failure"))

unmasked = results["random_unmasked"]["valid_action_rate"]
masked = results["random_masked"]["valid_action_rate"]
print("""
  without masking   %.1f%% of chosen actions are legal
  with masking      %.1f%%        <- the gate asks for >99%%

Masking is a structural guarantee, not a trained behaviour: an agent CANNOT emit
an illegal action, so this holds for every checkpoint Sudeepa produces. The
remaining %.1f%% in the unmasked row is what the temporal-baseline fallback
catches (PDF section 9.2) - the bundle is never dropped for a bad choice.
""" % (unmasked * 100, masked * 100, (1 - unmasked) * 100))

print(LINE)
print("READ THIS BEFORE TRAINING")
print(LINE)
print("""
The baseline scores %.3f on the primary metric. On stages 1-4 it scores exactly
1.000. There is almost no headroom for the RL agent to win, and that is a
scenario-design problem, not an agent problem.

Cause: one bundle per episode. With nothing else competing for contact capacity,
earliest-arrival IS optimal - there is nothing for a smarter policy to be
smarter about. Every PDF section 10.3 scenario where AI should win (congestion,
priority burst, combined stress) needs SEVERAL bundles contending for the same
finite windows.

Next task: cross-traffic. ScenarioConfig.bundles_per_episode already exists and
the environment currently ignores it.
""" % results["baseline"]["priority_weighted_timely"])
