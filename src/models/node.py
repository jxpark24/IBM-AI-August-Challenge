"""
Node = one physical asset in the network (PDF section 7.1 / Appendix A1).

Serafin owns the canonical Node model (PDF section 11.2). This file exists
because the RL environment needs node features TODAY and the interface freeze
was 20 August. It is written to the Appendix A1 schema exactly, so when the
integrated version lands it should be a drop-in replacement, not a rewrite.

The fixed 14 ids are a hard contract shared by the router, the environment and
the observation builder. Do not renumber them:

    0       science satellite      SCI
    1-8     LEO relays             LEO1..LEO8
    9-10    GEO relays             GEO1, GEO2
    11-13   ground stations        GNDA, GNDB, GNDC
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- the frozen id contract -------------------------------------------------

SCIENCE_ID = 0
LEO_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
GEO_IDS = (9, 10)
GROUND_IDS = (11, 12, 13)
NUM_NODES = 14  # == the RL action space size. Changing this breaks trained agents.

NODE_NAMES = {
    0: "SCI",
    1: "LEO1", 2: "LEO2", 3: "LEO3", 4: "LEO4",
    5: "LEO5", 6: "LEO6", 7: "LEO7", 8: "LEO8",
    9: "GEO1", 10: "GEO2",
    11: "GNDA", 12: "GNDB", 13: "GNDC",
}

# Link types, per PDF section 7.3.
OPTICAL_ISL = "OPTICAL_ISL"
RF_ISL = "RF_ISL"
RF_GROUND = "RF_GROUND"
OPTICAL_GROUND = "OPTICAL_GROUND"
GEO_RELAY = "GEO_RELAY"


@dataclass
class Terminal:
    """One radio or laser head. Terminal COUNT is what limits simultaneous links -
    a satellite with two terminals cannot talk to five neighbours at once."""

    terminal_id: str
    link_type: str
    data_rate_bps: float
    max_range_km: float = 0.0

    def __post_init__(self):
        if self.data_rate_bps <= 0:
            raise ValueError("data_rate_bps must be positive")


@dataclass
class Node:
    id: int
    type: str                                  # SCIENCE | LEO_RELAY | GEO_RELAY | GROUND
    name: str = ""

    position_xyz_km: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_xyz_km_s: Optional[Tuple[float, float, float]] = None

    # Storage-and-forward state. A bundle that cannot be stored is a bundle lost,
    # so free storage is a routing input, not bookkeeping.
    storage_capacity_bytes: int = 100 * 10**9
    storage_used_bytes: int = 0

    # Bytes already committed to outbound transmissions - the congestion signal
    # (PDF section 10.3: "one GEO or popular LEO relay queue at 85-95%").
    queue_bytes: int = 0

    health: float = 1.0        # 1.0 HEALTHY, 0.5 DEGRADED, 0.0 FAILED
    battery: float = 1.0       # energy proxy
    weather_risk: float = 0.0  # ground stations only; 0 CLEAR .. 1 BLOCKED

    terminals: List[Terminal] = field(default_factory=list)
    max_simultaneous_links: int = 2

    def __post_init__(self):
        if not 0 <= self.id < NUM_NODES:
            raise ValueError("node id must be in 0..%d" % (NUM_NODES - 1))
        for name, value in (("health", self.health),
                            ("battery", self.battery),
                            ("weather_risk", self.weather_risk)):
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in 0..1, got %r" % (name, value))
        if self.storage_capacity_bytes <= 0:
            raise ValueError("storage_capacity_bytes must be positive")
        if not 0 <= self.storage_used_bytes <= self.storage_capacity_bytes:
            raise ValueError("storage_used_bytes must be within 0..capacity")
        if self.queue_bytes < 0:
            raise ValueError("queue_bytes cannot be negative")
        if not self.name:
            self.name = NODE_NAMES.get(self.id, "N%d" % self.id)

    # --- derived features the observation builder reads ---------------------

    @property
    def is_ground(self) -> bool:
        return self.type == "GROUND"

    def storage_free_bytes(self) -> int:
        return self.storage_capacity_bytes - self.storage_used_bytes

    def storage_free_fraction(self) -> float:
        return self.storage_free_bytes() / self.storage_capacity_bytes

    def queue_fraction(self) -> float:
        """Backlog as a fraction of capacity, clipped to 1.0. Used directly as
        the congestion feature, so it must stay bounded even if a node is
        massively oversubscribed."""
        return min(1.0, self.queue_bytes / self.storage_capacity_bytes)

    def is_operational(self) -> bool:
        """A FAILED node cannot be routed through. DEGRADED still can - that is
        the point of giving the agent health as a continuous feature rather
        than a boolean."""
        return self.health > 0.0

    def can_store(self, size_bytes: int) -> bool:
        return self.storage_free_bytes() >= size_bytes


def default_terminals(node_type: str) -> List[Terminal]:
    """Prototype terminal fits. PROTOTYPE ASSUMPTION, not a real link budget -
    PDF section 15 requires these to be labelled as such. Harshul owns the real
    numbers; the ratios are what matter to the router (GEO slow but persistent,
    optical fast but weather-exposed)."""
    if node_type == "SCIENCE":
        return [
            Terminal("SCI-OPT", OPTICAL_ISL, 50e6, max_range_km=5000),
            Terminal("SCI-RF", RF_GROUND, 20e6, max_range_km=3000),
            Terminal("SCI-GEO", GEO_RELAY, 2e6, max_range_km=45000),
        ]
    if node_type == "LEO_RELAY":
        return [
            Terminal("LEO-OPT-A", OPTICAL_ISL, 50e6, max_range_km=5000),
            Terminal("LEO-OPT-B", OPTICAL_ISL, 50e6, max_range_km=5000),
            Terminal("LEO-GND", OPTICAL_GROUND, 100e6, max_range_km=2500),
        ]
    if node_type == "GEO_RELAY":
        return [
            Terminal("GEO-SPACE", GEO_RELAY, 2e6, max_range_km=45000),
            Terminal("GEO-GND", RF_GROUND, 2e6, max_range_km=45000),
        ]
    return [
        Terminal("GND-OPT", OPTICAL_GROUND, 100e6, max_range_km=2500),
        Terminal("GND-RF", RF_GROUND, 20e6, max_range_km=3000),
    ]


def build_default_nodes() -> Dict[int, Node]:
    """The 14-node network of PDF section 6: 1 science + 8 LEO + 2 GEO + 3 ground.

    Storage sizes are deliberately asymmetric: the science satellite has the
    small onboard buffer that creates the problem in the first place (PDF
    section 1.2, "onboard storage pressure"), ground stations are effectively
    infinite because delivery ends the bundle's life.
    """
    nodes: Dict[int, Node] = {}

    nodes[SCIENCE_ID] = Node(
        id=SCIENCE_ID, type="SCIENCE",
        storage_capacity_bytes=8 * 10**9,
        terminals=default_terminals("SCIENCE"),
        max_simultaneous_links=2,
    )
    for i in LEO_IDS:
        nodes[i] = Node(
            id=i, type="LEO_RELAY",
            storage_capacity_bytes=32 * 10**9,
            terminals=default_terminals("LEO_RELAY"),
            max_simultaneous_links=3,
        )
    for i in GEO_IDS:
        nodes[i] = Node(
            id=i, type="GEO_RELAY",
            storage_capacity_bytes=64 * 10**9,
            terminals=default_terminals("GEO_RELAY"),
            max_simultaneous_links=4,
        )
    for i in GROUND_IDS:
        nodes[i] = Node(
            id=i, type="GROUND",
            storage_capacity_bytes=10 * 10**12,
            terminals=default_terminals("GROUND"),
            max_simultaneous_links=4,
        )
    return nodes
