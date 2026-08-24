"""
Environment contract tests (PDF section 15: "fixed IDs + action masks +
invalid-action tests"; PDF section 12.2 gate for 24 August).

These are not "does RL work" tests - no agent is trained here. They pin down the
things that, if wrong, make every later training run silently meaningless:

    the mask and step() must agree about what is legal
    the observation must never change shape or leave [0, 1]
    the same seed must produce the same episode
    an illegal action must fall back, not crash and not drop the bundle
"""

import random

import pytest

from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.models.node import GROUND_IDS, NUM_NODES, SCIENCE_ID, build_default_nodes
from src.rl.candidates import enumerate_candidates
from src.rl.observation import CANDIDATE_FEATURES, OBS_SIZE
from src.rl.reward import DELIVERED, EXPIRED
from src.rl.routing_env import RoutingEnv
from src.rl.scenario import CURRICULUM, Scenario, ScenarioConfig, make_scenario

STAGES = sorted(CURRICULUM)


def rollout(env, policy, seed=None):
    """Run one episode, returning (trajectory, final info)."""
    obs, info = env.reset(seed=seed)
    trace = []
    for _ in range(64):
        action = policy(env)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((action, round(reward, 6), info["status"]))
        if terminated or truncated:
            break
    return trace, info


def baseline_policy(env):
    return env.baseline_action()


def masked_random_policy(rng):
    def policy(env):
        valid = [i for i, ok in enumerate(env.action_masks()) if ok]
        return rng.choice(valid)
    return policy


# --- observation contract ---------------------------------------------------

def test_observation_shape_is_frozen():
    """172 = 4 bundle + 14 one-hot + 14 nodes x 11 features. A checkpoint trained
    on one shape cannot be loaded against another, so this number is a contract."""
    assert OBS_SIZE == 4 + NUM_NODES + NUM_NODES * CANDIDATE_FEATURES == 172

    env = RoutingEnv(stage=2, seed=0)
    obs, _ = env.reset()
    assert len(obs) == OBS_SIZE


def test_observation_stays_normalised_across_every_stage():
    """Unbounded features diverge PPO value estimates. Check the hard cases -
    weather, failures and congestion all on."""
    for stage in STAGES:
        env = RoutingEnv(stage=stage, seed=stage)
        obs, _ = env.reset()
        for _ in range(8):
            assert all(0.0 <= float(x) <= 1.0 for x in obs), \
                "stage %d produced out-of-range observation" % stage
            action = env.baseline_action()
            if action is None:
                break
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break


def test_unreachable_nodes_are_zero_blocks():
    """Feature block j describes node j whether or not j is reachable. An
    unreachable node must be all zeros, starting with valid=0, so the policy can
    never confuse "no link" with "a link with small numbers"."""
    env = RoutingEnv(stage=2, seed=3)
    obs, _ = env.reset()
    mask = env.action_masks()

    for node_id in range(NUM_NODES):
        start = 4 + NUM_NODES + node_id * CANDIDATE_FEATURES
        block = [float(x) for x in obs[start:start + CANDIDATE_FEATURES]]
        if mask[node_id]:
            assert block[0] == 1.0
        else:
            assert block == [0.0] * CANDIDATE_FEATURES


# --- action mask contract ---------------------------------------------------

def test_mask_has_one_entry_per_node():
    env = RoutingEnv(stage=2, seed=0)
    env.reset()
    assert len(env.action_masks()) == NUM_NODES


def test_mask_is_never_all_false_while_the_episode_continues():
    """MaskablePPO cannot sample from an all-False mask - it raises. The
    environment must terminate the episode instead of ever presenting one."""
    rng = random.Random(0)
    policy = masked_random_policy(rng)

    for seed in range(30):
        env = RoutingEnv(stage=5, seed=seed)
        env.reset()
        for _ in range(16):
            assert any(env.action_masks()), \
                "seed %d presented an all-False mask" % seed
            _, _, terminated, truncated, _ = env.step(policy(env))
            if terminated or truncated:
                break


def test_every_masked_valid_action_is_actually_executable():
    """THE invariant. If the mask says yes and step() says no, training degrades
    with no error message anywhere."""
    for seed in range(25):
        env = RoutingEnv(stage=5, seed=seed)
        env.reset()
        for action in [i for i, ok in enumerate(env.action_masks()) if ok]:
            probe = RoutingEnv(stage=5, seed=seed)
            probe.reset()
            before = probe._now_s
            _, _, _, _, info = probe.step(action)
            assert info["invalid_actions"] == 0, \
                "mask allowed action %d but step() rejected it" % action
            assert probe._now_s >= before


def test_mask_encodes_physics_not_policy():
    """A bundle that cannot possibly make its deadline still has legal moves.
    Masking on "will miss the deadline" would empty the mask exactly when the
    agent most needs to learn something from the outcome."""
    plan = ContactPlan([
        Contact(SCIENCE_ID, 11, start_s=0, end_s=600, data_rate_bps=1e6),
    ])
    nodes = build_default_nodes()
    doomed = DataBundle("OBS-DOOM", SCIENCE_ID, size_bytes=125_000,
                        science_priority=0.9,
                        deadline_s=0.5)  # transfer alone takes 1.0 s

    scenario = Scenario(nodes=nodes, plan=plan, bundles=[doomed], horizon_s=600.0)
    env = RoutingEnv(scenario_factory=lambda **kw: scenario, seed=0)
    env.reset()

    assert env.action_masks()[11] is True
    _, reward, terminated, _, info = env.step(11)
    assert terminated
    # It arrived, just late. Late science is degraded, not worthless, so this is
    # DELIVERED-but-not-on-time and scores a small POSITIVE reward. Calling it
    # EXPIRED would make "deliver late" and "drop it" indistinguishable.
    assert info["status"] == DELIVERED
    assert info["on_time"] is False
    assert 0.0 < reward < 0.9


def test_bundle_stranded_past_its_deadline_expires():
    """The other half of the deadline story: EXPIRED is for data that never
    reached the ground, not for data that reached it late."""
    plan = ContactPlan([
        # The only link out of the science satellite opens long after the
        # deadline. Waiting for it is legal, and it strands the bundle at a
        # relay with the deadline already gone.
        Contact(SCIENCE_ID, 1, start_s=400, end_s=600, data_rate_bps=1e6),
    ])
    bundle = DataBundle("OBS-STRAND", SCIENCE_ID, size_bytes=125_000,
                        science_priority=0.8, deadline_s=200)
    scenario = Scenario(nodes=build_default_nodes(), plan=plan,
                        bundles=[bundle], horizon_s=600.0)

    env = RoutingEnv(scenario_factory=lambda **kw: scenario, seed=0)
    env.reset()

    assert env.action_masks()[1] is True             # still a legal move
    _, reward, terminated, _, info = env.step(1)     # waits until 400, arrives 401

    assert terminated
    assert info["status"] == EXPIRED                 # never reached the ground
    assert info["delivered"] is False
    assert reward < 0


# --- invalid actions and fallback -------------------------------------------

def test_invalid_action_falls_back_instead_of_dropping():
    """PDF section 9.2: "If invalid/unsafe/unavailable, call temporal baseline
    rather than drop the bundle"."""
    env = RoutingEnv(stage=2, seed=11)
    env.reset()

    invalid = [i for i, ok in enumerate(env.action_masks()) if not ok]
    if not invalid:
        pytest.skip("seed 11 had every node reachable")

    _, _, _, _, info = env.step(invalid[0])
    assert info["invalid_actions"] == 1
    assert info["fallbacks"] == 1
    assert info["status"] != "DROPPED"


def test_baseline_action_is_always_a_masked_valid_action():
    """The fallback must never propose something the mask forbids, or the safety
    net becomes the bug."""
    for seed in range(30):
        env = RoutingEnv(stage=5, seed=seed)
        env.reset()
        for _ in range(10):
            action = env.baseline_action()
            if action is None:
                break
            assert env.action_masks()[action], \
                "baseline proposed masked-out action %d (seed %d)" % (action, seed)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break


# --- determinism -------------------------------------------------------------

def test_same_seed_gives_an_identical_episode():
    """PDF section 10.5: every comparison uses identical traffic and seeds across
    policies. If this fails, no baseline-vs-AI number means anything."""
    a, info_a = rollout(RoutingEnv(stage=5), baseline_policy, seed=42)
    b, info_b = rollout(RoutingEnv(stage=5), baseline_policy, seed=42)

    assert a == b
    assert info_a["path"] == info_b["path"]
    assert info_a["latency_s"] == pytest.approx(info_b["latency_s"])


def test_different_seeds_give_different_episodes():
    a, _ = rollout(RoutingEnv(stage=5), baseline_policy, seed=1)
    b, _ = rollout(RoutingEnv(stage=5), baseline_policy, seed=2)
    assert a != b


# --- termination -------------------------------------------------------------

def test_delivery_terminates_the_episode():
    plan = ContactPlan([
        Contact(SCIENCE_ID, 11, start_s=0, end_s=600, data_rate_bps=1e6,
                propagation_delay_s=0.5),
    ])
    bundle = DataBundle("OBS-EASY", SCIENCE_ID, size_bytes=125_000,
                        science_priority=0.9, deadline_s=500)
    scenario = Scenario(nodes=build_default_nodes(), plan=plan,
                        bundles=[bundle], horizon_s=600.0)

    env = RoutingEnv(scenario_factory=lambda **kw: scenario, seed=0)
    env.reset()
    _, reward, terminated, truncated, info = env.step(11)

    assert terminated and not truncated
    assert info["status"] == DELIVERED and info["on_time"] is True
    assert info["latency_s"] == pytest.approx(1.5)
    assert reward > 0


def test_hop_count_is_bounded():
    """A ping-ponging policy must terminate rather than run forever."""
    rng = random.Random(7)
    policy = masked_random_policy(rng)
    for seed in range(20):
        env = RoutingEnv(stage=5, seed=seed, max_hops=4)
        trace, info = rollout(env, policy)
        assert info["hops"] <= 4
        assert len(trace) <= 64


# --- scenario / physics bookkeeping -----------------------------------------

def test_every_curriculum_stage_resets_and_runs():
    for stage in STAGES:
        env = RoutingEnv(stage=stage, seed=stage)
        obs, info = env.reset()
        assert len(obs) == OBS_SIZE
        assert any(env.action_masks())


def test_contacts_are_bidirectional():
    """A link you can only traverse one way is invisible until a route
    mysteriously fails, so assert it directly."""
    scenario = make_scenario(seed=0, stage=2)
    pairs = {(c.source_id, c.destination_id) for c in scenario.plan.contacts}
    missing = [(a, b) for (a, b) in pairs if (b, a) not in pairs]
    assert missing == []


def test_capacity_and_storage_are_consumed_on_a_hop():
    scenario = make_scenario(seed=5, stage=2)
    env = RoutingEnv(scenario_factory=lambda **kw: scenario, seed=0)
    env.reset()

    action = env.baseline_action()
    cand = env._candidates[action]
    capacity_before = cand.contact.residual_capacity_bytes
    size = env._bundle.remaining_bytes

    env.step(action)
    assert cand.contact.residual_capacity_bytes == capacity_before - size


def test_congestion_actually_binds():
    """PDF section 11.4 ablates congestion information. If congestion only
    decorated the feature vector the ablation would show nothing, for the wrong
    reason - so check it reaches the contacts."""
    clear = make_scenario(seed=3, config=ScenarioConfig(congestion=0.0))
    busy = make_scenario(seed=3, config=ScenarioConfig(congestion=0.9))

    def total_capacity(scn):
        return sum(c.residual_capacity_bytes for c in scn.plan.contacts)

    assert total_capacity(busy) < total_capacity(clear)


def test_failed_relay_has_no_contacts():
    for seed in range(40):
        scenario = make_scenario(seed=seed, stage=5)
        failed = [n.id for n in scenario.nodes.values() if n.health == 0.0]
        for node_id in failed:
            touching = [c for c in scenario.plan.contacts
                        if node_id in (c.source_id, c.destination_id)]
            assert touching == [], "failed node %d still has contacts" % node_id


def test_candidates_never_include_the_current_holder():
    """Self-hops would be a free infinite loop for a policy to exploit."""
    scenario = make_scenario(seed=2, stage=2)
    bundle = scenario.bundles[0]
    for holder in range(NUM_NODES):
        cands = enumerate_candidates(
            scenario.plan, scenario.nodes, holder, bundle.created_s,
            bundle.size_bytes, scenario.horizon_s)
        assert holder not in cands
