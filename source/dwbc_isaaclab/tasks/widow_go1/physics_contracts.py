"""Physics values that must remain identical to the frozen Isaac Gym task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionOffsets:
    """PhysX contact-generation distances, expressed in metres."""

    contact_offset: float
    rest_offset: float


def legacy_terrain_collision_offsets() -> CollisionOffsets:
    """Return the global offsets used when Isaac Gym created the triangle mesh."""
    return CollisionOffsets(contact_offset=0.01, rest_offset=0.0)
