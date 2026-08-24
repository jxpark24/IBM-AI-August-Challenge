"""
Mock dynamic scenario generator.

WHAT THIS IS NOT: orbital physics. There is no propagator here and there is not
meant to be one. Harshul and Serafin own the real contact generator (PDF section
8); when it lands it replaces THIS FILE ONLY, because everything downstream --
router, environment, observation builder -- consumes a ContactPlan and a dict of
Nodes and does not care where they came from.

WHAT THIS IS: the "random/mock dynamic graph with identical Node/Bundle
interface" that PDF section 11.6 says the RL workstream should start on so it is
not blocked on orbital integration. Every number in here is a PROTOTYPE
ASSUMPTION (PDF section 15) chosen to make the routing PROBLEM realistic, not to
claim the geometry is real:

  - GEO is always reachable but slow            -> the tempting-but-wrong path
  - LEO passes are short and intermittent       -> waiting is mandatory
  - direct-to-ground is rare and fast           -> the ESA ~10-in-100 motif [1]
  - optical ground links die in bad weather     -> RF is slower but survives

The curriculum stages implement PDF section 9.4's risk-minimising training
sequence. Sudeepa walks up the stages; the environment interface never changes.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.models.node import (
    GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_ID,
    GEO_RELAY, OPTICAL_GROUND, OPTICAL_ISL, RF_GROUND,
    Node, build_default_nodes,
)

# --- prototype link assumptions (bits/s) ------------------------------------
RATE_GEO = 2e6            # persistent but slow
RATE_ISL = 50e6           # LEO crosslink, optical
RATE_GND_OPTICAL = 100e6  # fast, weather-exposed
RATE_GND_RF = 20e6        # slower, weather-tolerant

PROP_GEO = 0.12           # ~36000 km / c
PROP_ISL = 0.005
PROP_GND = 0.004


@dataclass
class ScenarioConfig:
    """One knob per PDF section 9.4 curriculum step, so difficulty is explicit
    rather than hidden in magic numbers."""

    horizon_s: float = 3600.0
    static: bool = False            # stage 1: contacts open for the whole horizon
    enable_priorities: bool = True  # stage 4
    congestion: float = 0.0         # stage 3: 0..1 fraction of relay capacity pre-used
    enable_weather: bool = False    # stage 5
    enable_failures: bool = False   # stage 5
    bundles_per_episode: int = 1


# PDF section 9.4: "make the agent succeed on a static toy graph" first, then add
# one source of difficulty at a time. Held-out evaluation uses unseen SEEDS on
# these same stages, plus stage 5 configurations never trained on.
CURRICULUM = {
    1: ScenarioConfig(static=True, enable_priorities=False),
    2: ScenarioConfig(),
    3: ScenarioConfig(congestion=0.6),
    4: ScenarioConfig(congestion=0.6, enable_priorities=True),
    5: ScenarioConfig(congestion=0.6, enable_priorities=True,
                      enable_weather=True, enable_failures=True),
}


@dataclass
class Scenario:
    """Everything one episode needs. Deliberately a plain data object: the
    environment should be able to accept a scenario built by the real orbital
    contact generator with no code change."""

    nodes: Dict[int, Node]
    plan: ContactPlan
    bundles: List[DataBundle]
    horizon_s: float
    ground_ids: Tuple[int, ...] = GROUND_IDS
    notes: List[str] = field(default_factory=list)


def _windows(rng, horizon_s, period_s, duration_s, jitter=0.25):
    """Periodic visibility windows with a random phase - the shape of a satellite
    pass, without pretending to be one. Returns [(start, end), ...]."""
    out = []
    phase = rng.uniform(0.0, period_s)
    t = phase - period_s
    while t < horizon_s:
        dur = duration_s * (1.0 + rng.uniform(-jitter, jitter))
        start, end = max(0.0, t), min(horizon_s, t + dur)
        if end - start > 1.0:
            out.append((start, end))
        t += period_s * (1.0 + rng.uniform(-jitter, jitter))
    return out


def _pair_one(contacts, src, dst, start, end, rate, prop, link_type,
              capacity, reliability=1.0, weather=0.0):
    """Append ONE direction. Callers add both. The router indexes contacts by
    source, so a link you can only traverse one way is a bug that is very easy
    to create here -- and it is invisible until a route mysteriously fails."""
    contacts.append(Contact(
        source_id=src, destination_id=dst,
        start_s=start, end_s=end,
        data_rate_bps=rate, propagation_delay_s=prop,
        residual_capacity_bytes=capacity,
        reliability=reliability, weather_risk=weather,
        link_type=link_type,
        energy_cost=0.8 if link_type == GEO_RELAY else 0.3,
    ))


def make_scenario(seed=None, config=None, stage=None) -> Scenario:
    """Build one seeded episode. Same seed + same config => byte-identical
    scenario (PDF section 10.5 requires identical traffic across policies)."""
    if stage is not None:
        config = CURRICULUM[stage]
    config = config or ScenarioConfig()
    rng = random.Random(seed)

    nodes = build_default_nodes()
    horizon = config.horizon_s
    notes: List[str] = []
    contacts: List[Contact] = []

    big = 10**15  # effectively uncapped contact capacity

    # --- weather (PDF section 8.3): optical dies, RF survives ---------------
    blocked_ground = set()
    if config.enable_weather:
        for gid in GROUND_IDS:
            roll = rng.random()
            if roll < 0.20:
                nodes[gid].weather_risk = 1.0
                blocked_ground.add(gid)
                notes.append("%s BLOCKED by weather" % nodes[gid].name)
            elif roll < 0.45:
                nodes[gid].weather_risk = rng.uniform(0.3, 0.7)
                notes.append("%s DEGRADED (risk %.2f)"
                             % (nodes[gid].name, nodes[gid].weather_risk))

    # --- relay failure (PDF section 10.3) ----------------------------------
    failed_leo = None
    if config.enable_failures and rng.random() < 0.5:
        failed_leo = rng.choice(LEO_IDS)
        nodes[failed_leo].health = 0.0
        notes.append("%s FAILED" % nodes[failed_leo].name)

    # --- congestion (PDF section 10.3): pre-load relay queues ---------------
    # This must BIND physically, not just show up in the feature vector. A queue
    # number the router can ignore teaches the agent nothing, and the section
    # 11.4 ablation ("remove congestion information") would show no effect for
    # the wrong reason. So a congested node's outbound contacts lose capacity
    # AND lose effective rate - queueing delay behind other traffic.
    crowded = set()
    if config.congestion > 0:
        crowded = set(rng.sample(list(LEO_IDS), 2) + [rng.choice(GEO_IDS)])
        for nid in crowded:
            frac = config.congestion * rng.uniform(0.85, 0.95)
            nodes[nid].queue_bytes = int(nodes[nid].storage_capacity_bytes * frac)
            nodes[nid].storage_used_bytes = nodes[nid].queue_bytes
        notes.append("congested: %s"
                     % ", ".join(sorted(nodes[n].name for n in crowded)))

    def _derate(src, rate, cap):
        """A backed-up relay forwards slower and has less of the window left."""
        if src not in crowded:
            return rate, cap
        share = 1.0 - nodes[src].queue_fraction()
        return rate * max(0.15, share), int(cap * max(0.2, share))

    def add(a, b, wins, rate, prop, link_type, cap=big, rel=1.0, weather=0.0):
        if config.static:
            wins = [(0.0, horizon)]
        for start, end in wins:
            for src, dst in ((a, b), (b, a)):
                r, c = _derate(src, rate, cap)
                _pair_one(contacts, src, dst, start, end, r, prop, link_type,
                          c, rel, weather)

    # --- GEO: persistent, slow, the plausible-looking trap ------------------
    for gid in GEO_IDS:
        add(SCIENCE_ID, gid, [(0.0, horizon)], RATE_GEO, PROP_GEO, GEO_RELAY)
        for lid in LEO_IDS:
            add(lid, gid, [(0.0, horizon)], RATE_GEO, PROP_GEO, GEO_RELAY)
        for gnd in GROUND_IDS:
            add(gid, gnd, [(0.0, horizon)], RATE_GEO, PROP_GEO, RF_GROUND)

    # --- science satellite -> LEO relays: short intermittent passes ---------
    for lid in LEO_IDS:
        add(SCIENCE_ID, lid,
            _windows(rng, horizon, period_s=900, duration_s=150),
            RATE_ISL, PROP_ISL, OPTICAL_ISL)

    # --- LEO crosslinks: ring neighbours only, so multi-hop actually means
    #     something and the mesh is not a fully connected cheat ---------------
    for k, lid in enumerate(LEO_IDS):
        nxt = LEO_IDS[(k + 1) % len(LEO_IDS)]
        add(lid, nxt,
            _windows(rng, horizon, period_s=700, duration_s=180),
            RATE_ISL, PROP_ISL, OPTICAL_ISL)

    # --- LEO -> ground gateways --------------------------------------------
    for lid in LEO_IDS:
        for gnd in GROUND_IDS:
            wins = _windows(rng, horizon, period_s=1300, duration_s=200)
            risk = nodes[gnd].weather_risk
            if gnd in blocked_ground:
                # Optical gateway is out. RF still works: the robust-but-slower
                # alternative PDF section 8.3 wants the agent to discover.
                add(lid, gnd, wins, RATE_GND_RF, PROP_GND, RF_GROUND, rel=0.95)
            else:
                add(lid, gnd, wins, RATE_GND_OPTICAL, PROP_GND, OPTICAL_GROUND,
                    rel=1.0 - 0.5 * risk, weather=risk)

    # --- direct-to-ground: rare and fast (the ESA ~10-in-100 motif) ---------
    for gnd in GROUND_IDS:
        # ~3% duty per station, ~10% across the three -- the ESA "10 minutes in
        # every 100" shorthand [1], which is the whole reason relays exist.
        wins = _windows(rng, horizon, period_s=3000, duration_s=100)
        if gnd in blocked_ground:
            add(SCIENCE_ID, gnd, wins, RATE_GND_RF, PROP_GND, RF_GROUND, rel=0.95)
        else:
            add(SCIENCE_ID, gnd, wins, RATE_GND_OPTICAL, PROP_GND,
                OPTICAL_GROUND, rel=1.0 - 0.5 * nodes[gnd].weather_risk,
                weather=nodes[gnd].weather_risk)

    # A failed relay's links simply do not exist.
    if failed_leo is not None:
        contacts = [c for c in contacts
                    if c.source_id != failed_leo and c.destination_id != failed_leo]

    plan = ContactPlan(contacts)
    bundles = [_make_bundle(rng, i, horizon, config)
               for i in range(config.bundles_per_episode)]

    return Scenario(nodes=nodes, plan=plan, bundles=bundles,
                    horizon_s=horizon, notes=notes)


# Science traffic classes (PDF section 7.4 / section 11.1). Relative priorities
# and deadlines are SIMULATION ASSUMPTIONS, not universal mission requirements.
#   (data_type, priority range, size MB range, deadline seconds range)
TRAFFIC_MIX = [
    ("TRANSIENT",    (0.88, 1.00), (40, 160),   (240, 900)),
    ("STAR_FIELD",   (0.45, 0.70), (300, 900),  (1200, 3000)),
    ("CALIBRATION",  (0.20, 0.40), (80, 300),   (1800, 3400)),
    ("HOUSEKEEPING", (0.05, 0.15), (5, 40),     (2400, 3500)),
]


def _make_bundle(rng, index, horizon, config) -> DataBundle:
    if config.enable_priorities:
        data_type, pri, size_mb, dl = rng.choice(TRAFFIC_MIX)
    else:
        # Stage 1-3: every bundle identical, so any behaviour difference the
        # agent shows is about ROUTING, not about priority.
        data_type, pri, size_mb, dl = ("STAR_FIELD", (0.5, 0.5), (100, 100),
                                       (horizon, horizon))

    # An observation does not politely arrive at t=0. Starting episodes at a
    # random moment stops the agent memorising "the window that is open at zero".
    created = 0.0 if config.static else rng.uniform(0.0, horizon * 0.45)
    return DataBundle(
        bundle_id="OBS-%06d" % (rng.randrange(1, 999999)),
        source_id=SCIENCE_ID,
        size_bytes=int(rng.uniform(*size_mb) * 10**6),
        created_s=created,
        science_priority=round(rng.uniform(*pri), 3),
        deadline_s=min(horizon, created + rng.uniform(*dl)),
        data_type=data_type,
    )
