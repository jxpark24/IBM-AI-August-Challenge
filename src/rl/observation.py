"""
Fixed-size observation builder (PDF section 9.3 / Appendix A2).

OWNERSHIP: Sudeepa owns feature normalisation and tuning (PDF section 11.4).
Jiwoo owns the SHAPE and the ORDER, because those are the frozen interface -
change them and every trained checkpoint becomes garbage.

    [0:4]     bundle features
    [4:18]    one-hot of the node currently holding the bundle (14)
    [18:172]  14 candidates x 11 features, indexed by NODE ID not by rank

That last point matters. Feature block j always describes node j, whether or not
node j is reachable; unreachable nodes are all zeros with valid=0. A "top-5
reachable neighbours" layout would be smaller, but the meaning of each slot
would shift between steps and the policy could never learn "GEO2 is slow".

Every value is clipped to [0, 1]. Not for elegance - unbounded features make PPO
value estimates diverge, and the normalising constants below are prototype
guesses that WILL be exceeded.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.models.bundle import DataBundle
from src.models.contact import ContactPlan
from src.models.node import GROUND_IDS, NUM_NODES, Node
from src.rl.candidates import CandidateLink
from src.routing.temporal_baseline import earliest_arrival

BUNDLE_FEATURES = 4
CANDIDATE_FEATURES = 11
OBS_SIZE = BUNDLE_FEATURES + NUM_NODES + NUM_NODES * CANDIDATE_FEATURES  # 172

# Human-readable names, in order. Used by tests and by the model card so nobody
# has to count array offsets to find out what index 93 means.
CANDIDATE_FEATURE_NAMES = (
    "valid",
    "link_rate_norm",
    "contact_remaining_norm",
    "prop_delay_norm",
    "queue_norm",
    "storage_free_norm",
    "health",
    "battery",
    "weather_risk",
    "estimated_arrival_to_ground_norm",
    "estimated_route_reliability",
)


@dataclass
class Normaliser:
    """PROTOTYPE ASSUMPTIONS. Sudeepa's to tune; they must be FROZEN alongside
    the checkpoint (PDF section 10.5) or replaying a saved agent silently feeds
    it differently-scaled inputs."""

    max_size_bytes: float = 1e9
    max_rate_bps: float = 100e6
    max_prop_s: float = 0.5
    horizon_s: float = 3600.0
    max_deadline_s: float = 3600.0


def _clip01(x: float) -> float:
    if x != x:          # NaN guard: one NaN poisons the whole forward pass
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def build_observation(
    bundle: DataBundle,
    holder_id: int,
    now_s: float,
    nodes: Dict[int, Node],
    plan: ContactPlan,
    candidates: Dict[int, CandidateLink],
    norm: Optional[Normaliser] = None,
) -> List[float]:
    """Assemble the observation. `candidates` comes from the environment so the
    observation and the action mask can never disagree about what is reachable."""
    norm = norm or Normaliser()
    obs: List[float] = []

    # --- bundle: what is being carried and how urgent it is -----------------
    deadline_left = (bundle.deadline_s - now_s) if bundle.deadline_s is not None \
        else norm.max_deadline_s
    obs.append(_clip01(bundle.science_priority))
    obs.append(_clip01(bundle.remaining_bytes / norm.max_size_bytes))
    obs.append(_clip01(deadline_left / norm.max_deadline_s))
    obs.append(_clip01((now_s - bundle.created_s) / norm.horizon_s))

    # --- where it is now ----------------------------------------------------
    obs.extend(1.0 if i == holder_id else 0.0 for i in range(NUM_NODES))

    # --- one block per node id ----------------------------------------------
    for node_id in range(NUM_NODES):
        cand = candidates.get(node_id)
        if cand is None:
            obs.extend([0.0] * CANDIDATE_FEATURES)
            continue

        node = nodes[node_id]
        contact = cand.contact
        arrival_norm, reliability = _lookahead(
            plan, node_id, cand.arrival_s, bundle, norm)

        obs.extend([
            1.0,
            _clip01(contact.data_rate_bps / norm.max_rate_bps),
            _clip01(cand.window_remaining_s / norm.horizon_s),
            _clip01(contact.propagation_delay_s / norm.max_prop_s),
            _clip01(node.queue_fraction()),
            _clip01(node.storage_free_fraction()),
            _clip01(node.health),
            _clip01(node.battery),
            _clip01(max(node.weather_risk, contact.weather_risk)),
            arrival_norm,
            reliability,
        ])

    assert len(obs) == OBS_SIZE, "observation shape drifted: %d != %d" % (
        len(obs), OBS_SIZE)
    return obs


def _lookahead(plan, node_id, arrival_s, bundle, norm):
    """How good does the rest of the journey look from this candidate?

    This runs the temporal baseline forward from the candidate node. It is by
    far the most expensive feature and it is also the one that makes the agent's
    job tractable: without it the policy would have to rediscover contact-graph
    search from reward alone, which is not going to happen in the time available.

    Giving the agent the baseline's own estimate is deliberate. The agent is not
    being asked to beat the baseline at finding the earliest arrival - it is
    being asked to know WHEN earliest arrival is the wrong objective, because of
    congestion, weather, reliability or a deadline that makes a slower-but-safer
    path better (PDF section 10.3).

    Returns (arrival_norm, reliability). arrival_norm is 1.0 for "no route from
    here", which reads as maximally-late and is exactly the signal we want.
    """
    if node_id in GROUND_IDS:
        return 0.0, 1.0  # already delivered; nothing left to traverse

    route = earliest_arrival(
        plan, node_id, GROUND_IDS, bundle.remaining_bytes,
        start_s=arrival_s, deadline_s=None,
    )
    if route is None:
        return 1.0, 0.0

    arrival_norm = _clip01(route.arrival_s / norm.horizon_s)
    reliability = 1.0
    for hop in route.hops:
        reliability *= hop.reliability
    return arrival_norm, _clip01(reliability)
