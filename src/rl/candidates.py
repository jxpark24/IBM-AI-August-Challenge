"""
What counts as a legal next hop.

This module exists to kill one specific bug class. The action mask, the
observation builder and step() all need to answer "can the bundle go from here
to node j right now?". If three copies of that logic exist they WILL drift, and
the symptom is the worst kind: the agent picks an action the mask said was
valid, step() refuses it, and the training curve quietly degrades with no error.

So the answer is computed once, here, and the environment passes the same
CandidateLink objects to all three.

Design rule: this encodes PHYSICS, not POLICY.
    physics -> the window is shut, the transfer cannot finish, the relay is dead
    policy  -> this route will miss the deadline, this relay looks congested

Only physics masks an action out. "This will miss the deadline" stays legal and
scores badly, because the agent has to LEARN that, and because a mask driven by
the deadline goes all-False the moment a bundle is doomed - which crashes
MaskablePPO instead of teaching it anything.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from src.models.contact import Contact, ContactPlan
from src.models.node import Node


@dataclass
class CandidateLink:
    """One executable next hop, with the timing already worked out."""

    dest_id: int
    contact: Contact
    now_s: float       # when the decision is being made
    depart_s: float    # when transmission starts (>= now; waiting is legal)
    tx_s: float        # how long the transfer takes
    arrival_s: float   # depart + tx + propagation

    @property
    def wait_s(self) -> float:
        """Seconds the bundle sits in storage before this hop departs. Non-zero
        means the agent chose to WAIT for a window - the behaviour a snapshot
        router cannot express at all."""
        return max(0.0, self.depart_s - self.now_s)

    @property
    def window_remaining_s(self) -> float:
        """Usable window left from now. Feeds contact_remaining_norm."""
        return max(0.0, self.contact.end_s - self.depart_s)


def enumerate_candidates(
    plan: ContactPlan,
    nodes: Dict[int, Node],
    holder_id: int,
    now_s: float,
    size_bytes: int,
    horizon_s: float,
) -> Dict[int, CandidateLink]:
    """All legal next hops from holder_id at now_s, best contact per destination.

    "Best" = earliest arrival, which is the same tie-break the temporal baseline
    uses, so the agent and the benchmark are choosing between the same options.
    """
    out: Dict[int, CandidateLink] = {}

    for contact in plan.from_node(holder_id):
        dest = contact.destination_id
        if dest == holder_id:
            continue

        node = nodes.get(dest)
        if node is None or not node.is_operational():
            continue

        # 1. Wait for the window if it has not opened (store-and-forward).
        depart_s = max(now_s, contact.start_s)
        if depart_s >= contact.end_s:
            continue  # 2. window already closed

        # 3. The transfer has to finish before the window shuts.
        tx_s = contact.transmission_time_s(size_bytes)
        if depart_s + tx_s > contact.end_s:
            continue

        # 4. Contact capacity left after other traffic.
        if contact.residual_capacity_bytes < size_bytes:
            continue

        # 5. The receiver has to be able to hold it. Ground stations are the
        #    exception: delivery ends the bundle's life, nothing is stored.
        if not node.is_ground and not node.can_store(size_bytes):
            continue

        arrival_s = depart_s + tx_s + contact.propagation_delay_s
        if arrival_s > horizon_s:
            continue  # 6. lands after the scenario ends

        best = out.get(dest)
        if best is None or arrival_s < best.arrival_s:
            out[dest] = CandidateLink(dest, contact, now_s, depart_s, tx_s, arrival_s)

    return out
