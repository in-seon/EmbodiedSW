from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Occupant:
    progress: int
    eta_sec: Optional[float] = None


def _as_occupant(item) -> Occupant:
    if isinstance(item, Occupant):
        return item
    return Occupant(progress=item)


def minimum_progress(occupants) -> Optional[int]:
    values = [
        person.progress
        for person in (_as_occupant(o) for o in occupants if o is not None)
        if person.progress is not None
    ]
    return min(values) if values else None


def maximum_eta_sec(occupants) -> Optional[float]:
    values = [
        person.eta_sec
        for person in (_as_occupant(o) for o in occupants if o is not None)
        if person.eta_sec is not None
    ]
    return max(values) if values else None
