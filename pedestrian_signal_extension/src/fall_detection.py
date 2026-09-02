from dataclasses import dataclass
from collections import deque

import numpy as np

from config import config

@dataclass
class Person:
    bbox: tuple        
    conf: float
    keypoints: np.ndarray | None = None  
    track_id: int | None = None         

def foot_in_roi(person: Person, roi_px) -> bool:
    x1, y1, x2, y2 = person.bbox
    foot_x = (x1 + x2) / 2
    foot_y = y2 - config.FOOT_POINT_OFFSET_RATIO * (y2 - y1)
    rx1, ry1, rx2, ry2 = roi_px
    return rx1 <= foot_x <= rx2 and ry1 <= foot_y <= ry2


def roi_overlap_ratio(person: Person, roi_px) -> float:
    x1, y1, x2, y2 = person.bbox
    rx1, ry1, rx2, ry2 = roi_px
    iw = max(0, min(x2, rx2) - max(x1, rx1))
    ih = max(0, min(y2, ry2) - max(y1, ry1))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (iw * ih) / area if area > 0 else 0.0


def bbox_overlap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    smaller = min(max(0, ax2 - ax1) * max(0, ay2 - ay1),
                  max(0, bx2 - bx1) * max(0, by2 - by1))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def torso_angle_deg(kp: np.ndarray) -> float | None:
    if kp is None:
        return None
    sh = kp[[5, 6]]
    hip = kp[[11, 12]]
    if sh[:, 2].min() < 0.3 or hip[:, 2].min() < 0.3:
        return None
    v = hip[:, :2].mean(axis=0) - sh[:, :2].mean(axis=0)
    if np.linalg.norm(v) < 1e-6:
        return None
    cos = abs(v[1]) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))))


def looks_fallen(person: Person, cfg) -> bool:
    ang = torso_angle_deg(person.keypoints)
    if ang is not None:
        return ang > cfg["fall_angle_deg"]
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    return h > 0 and (w / h) > cfg["fall_aspect_ratio"]


class FallMonitor:

    def __init__(self, cfg):
        self.cfg = cfg
        self.candidate_since: float | None = None  
        self.last_fallen: float | None = None      
        self.clear_since: float | None = None     
        self.confirmed = False

    def update(self, any_fallen: bool, now: float, gap_sec: float) -> bool:
        if any_fallen:
            self.last_fallen = now
            self.clear_since = None
            if self.candidate_since is None:
                self.candidate_since = now
            if now - self.candidate_since >= self.cfg["fall_confirm_sec"]:
                self.confirmed = True
        elif self.confirmed:
            self.candidate_since = None
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= self.cfg["fall_clear_sec"]:
                self.confirmed = False
                self.clear_since = None
        else:
            self.clear_since = None
            if (self.candidate_since is not None and self.last_fallen is not None
                    and now - self.last_fallen > gap_sec):
                self.candidate_since = None
        return self.confirmed


class FallTracker:

    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors: dict[int, FallMonitor] = {}
        self.last_seen: dict[int, float] = {}
        self.last_bbox: dict[int, tuple] = {}
        self.intervals = deque(maxlen=15)   # 실측 프레임 간격 (중앙값으로 갭 산출)
        self.prev_now: float | None = None
        self._synth = 0                     # ID 없는 검출용 임시 키 (음수, 트래커 ID와 안 겹침)

    def _tick(self, now: float) -> float | None:
        if self.prev_now is not None and now > self.prev_now:
            self.intervals.append(now - self.prev_now)
        self.prev_now = now
        if not self.intervals:
            return None
        return sorted(self.intervals)[len(self.intervals) // 2]

    def gap_sec(self, med: float | None) -> float:
        if med is None:
            return self.cfg["fall_gap_sec"]
        return min(max(self.cfg["fall_gap_sec"], med * self.cfg["fall_gap_frames"]),
                   self.cfg["fall_gap_max_sec"])

    def grace_sec(self, med: float | None) -> float:
        if med is None:
            return self.cfg["track_grace_sec"]
        return min(max(self.cfg["track_grace_sec"], med * self.cfg["track_grace_frames"]),
                   self.cfg["track_grace_max_sec"])

    def _match(self, bbox, now, taken, grace: float) -> int | None:
        best, best_ov = None, self.cfg["track_inherit_overlap"]
        for key in self.monitors:
            if key in taken:                     
                continue
            if key not in self.last_bbox or key not in self.last_seen:
                continue
            if now - self.last_seen[key] > grace:
                continue
            ov = bbox_overlap(self.last_bbox[key], bbox)
            if ov >= best_ov:
                best, best_ov = key, ov
        return best

    def _rekey(self, old: int, new: int) -> int:
        if new in self.monitors:  
            return old
        self.monitors[new] = self.monitors.pop(old)
        if old in self.last_seen:
            self.last_seen[new] = self.last_seen.pop(old)
        if old in self.last_bbox:
            self.last_bbox[new] = self.last_bbox.pop(old)
        return new

    def _resolve_keys(self, people, now, grace: float) -> list[int]:
        keys: list[int | None] = [None] * len(people)
        taken: set[int] = set()

        for i, p in enumerate(people):
            if p.track_id is not None and p.track_id in self.monitors:
                keys[i] = p.track_id
                taken.add(p.track_id)

        for i, p in enumerate(people):
            if keys[i] is not None:
                continue
            src = self._match(p.bbox, now, taken, grace)
            if src is None:                     
                if p.track_id is not None:
                    keys[i] = p.track_id
                else:
                    self._synth -= 1
                    keys[i] = self._synth
            elif p.track_id is None:
                keys[i] = src                    
            else:
                keys[i] = self._rekey(src, p.track_id)   
            taken.add(keys[i])
        return keys

    def _retire(self, now: float, gap: float, grace: float):
        ttl = self.cfg["fall_clear_sec"] + gap + grace
        stale = [t for t, m in self.monitors.items()
                 if not m.confirmed and (t not in self.last_seen
                                         or now - self.last_seen[t] > ttl)]
        for tid in stale:
            del self.monitors[tid]
            self.last_seen.pop(tid, None)
            self.last_bbox.pop(tid, None)

    def update(self, people, fallen_flags, on_cw_flags, now: float) -> set[int]:
        med = self._tick(now)         
        gap, grace = self.gap_sec(med), self.grace_sec(med)
        keys = self._resolve_keys(people, now, grace)

        seen: dict[int, bool] = {}
        for p, key, fallen, on_cw in zip(people, keys, fallen_flags, on_cw_flags):
            p.track_id = key
            seen[key] = seen.get(key, False) or (fallen and on_cw)
            self.last_bbox[key] = p.bbox

        for key, fallen in seen.items():
            self.monitors.setdefault(key, FallMonitor(self.cfg)).update(fallen, now, gap)
            self.last_seen[key] = now

        for key in [k for k in self.monitors if k not in seen]:
            self.monitors[key].update(False, now, gap)

        self._retire(now, gap, grace)
        return {k for k, m in self.monitors.items() if m.confirmed}

    def reset(self):
        self.monitors.clear()
        self.last_seen.clear()
        self.last_bbox.clear()

def person_from_box(box) -> Person:
    return Person(
        bbox=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
        conf=float(box.confidence),
        keypoints=getattr(box, "keypoints", None),
        track_id=box.track_id,
    )


def people_from_boxes(boxes) -> list:
    return [person_from_box(b) for b in boxes if b.is_pedestrian()]


def roi_from_ratio(frame_shape, ratio=None):
    ratio = ratio if ratio is not None else config.FALL_CONFIG["crosswalk_roi"]
    h, w = frame_shape[0], frame_shape[1]
    return (int(ratio[0] * w), int(ratio[1] * h), int(ratio[2] * w), int(ratio[3] * h))


def roi_from_zones(zones):
    xs, ys = [], []
    for zone in zones.zones:
        for x, y in zone.points:
            xs.append(x)
            ys.append(y)
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


class FallDetectionPipeline:
    def __init__(self, roi_px, cfg=None):
        self.cfg = cfg if cfg is not None else config.FALL_CONFIG
        self.roi_px = roi_px
        self.tracker = FallTracker(self.cfg)

    def update(self, boxes, now) -> dict:
        people = people_from_boxes(boxes)
        in_roi_flags = [foot_in_roi(p, self.roi_px) for p in people]
        fallen_flags = [looks_fallen(p, self.cfg) for p in people]
        on_cw_flags = [
            roi_overlap_ratio(p, self.roi_px) >= self.cfg["fall_roi_overlap"]
            for p in people
        ]
        confirmed_ids = self.tracker.update(people, fallen_flags, on_cw_flags, now)
        return {
            "people": people,
            "fallen_flags": fallen_flags,
            "in_roi_flags": in_roi_flags,
            "confirmed_ids": confirmed_ids,
            "fall_confirmed": bool(confirmed_ids),
        }

    def reset(self):
        self.tracker.reset()
