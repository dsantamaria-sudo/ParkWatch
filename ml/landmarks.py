"""ParkWatch — camera landmarks: anchors from Label Studio, tracking photo by photo, spot warps.

The camera is glued in place and moves (slow drift, jumps, a fall). Fixed points of the scene
(landmarks lm1..lmN, project parkwatch-landmarks, label_studio/landmarks_config.xml) are
clicked once on the reference photo of rois.json and, now and then, on another photo (an
ANCHOR: the exact camera position for that photo). Every other photo is tracked:

  - each landmark is searched near where it was in the previous photo (normalised
    cross-correlation of a patch cut from the anchor nearest in time);
  - a match is kept if it agrees with the rigid movement (rotation + shift) of the others
    since the previous photo; the rotation is only estimated from matches spread out enough;
  - a landmark not found is moved with that common movement (so every spot keeps its
    landmarks around it) and, after MISSED_WIDE photos, searched in a wider radius;
  - with fewer than MIN_LANDMARKS matches, or an implausible rotation since the previous photo
    (above MAX_STEP_ROTATION_DEG, or above BIG_STEP_ROTATION_DEG with fewer than
    BIG_STEP_MIN_LANDMARKS matches: chance matches), the photo is LOST: the previous positions
    are kept.

The lens is a fisheye and the camera does not only rotate: every part of the image moves
differently. Each spot therefore gets its own warp (reference -> photo), a similarity fitted
to the landmarks weighted by their distance to the spot (local_warp).

Validated on 16/10-7/11 with 12 anchors: spot centres within 10 px of the clicks (mostly
1-6 px). Used by align_photos.py (pipeline) and by tools/landmarks_from_ls.py and
tools/landmark_tracking.py (local analysis).
"""
from __future__ import annotations

import bisect
import math
import re
from collections.abc import Callable
from datetime import datetime

import cv2
import numpy as np

Warp = tuple[float, float, float, float, float, float]  # m00 m01 m02 / m10 m11 m12 (reference -> photo)
Points = dict[str, np.ndarray]                         # landmark -> (x, y) in px

SIGMA_PX = 200.0        # local model: landmark weight exp(-d^2 / 2 sigma^2), d = distance (best of 150-500 on 6/11)
OUTLIER_PX = 60.0       # an anchor click this far from what the other landmarks predict is a misclick
PATCH = 40              # half size of the template patch (81x81 px)
RADIUS = 40             # search radius around the previous position (px)
RADIUS_LOST = 120       # wider search after a lost photo or for a landmark missed MISSED_WIDE times
MIN_SCORE = 0.5         # normalised cross-correlation
MAX_DISAGREE = 8        # px: a match further than this from the rigid movement of the others is rejected
SPREAD_ROTATION = 400   # px: the rotation between photos is only estimated from matches this spread out
MISSED_WIDE = 3
MIN_LANDMARKS = 3
# Implausible movement between two photos: chance matches in the dark (9/11 06:08: 3 false
# matches, -20 deg, every landmark lost for the rest of the day). The real overnight moves seen
# (4/11) turn 3-3.5 deg with 5-8 matches. A larger rotation is never accepted, a big one only
# with enough matches; otherwise the photo is LOST (a real knock is fixed with an anchor).
MAX_STEP_ROTATION_DEG = 10.0
BIG_STEP_ROTATION_DEG = 2.0
BIG_STEP_MIN_LANDMARKS = 5

PHOTO_RE = re.compile(r"parkwatch_(\d{4})(\d{2})(\d{2})T\d{6}Z\.jpg$")


# --- geometry -------------------------------------------------------------------------------

def apply(warp: Warp, pts: np.ndarray) -> np.ndarray:
    m00, m01, m02, m10, m11, m12 = warp
    return np.stack([m00 * pts[:, 0] + m01 * pts[:, 1] + m02, m10 * pts[:, 0] + m11 * pts[:, 1] + m12], axis=1)


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[Warp, np.ndarray]:
    """Least-squares rotation + shift mapping src onto dst; the warp and the residual of each point."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - ms, dst - md
    theta = math.atan2((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum(), (s * d).sum())
    c, sn = math.cos(theta), math.sin(theta)
    tx, ty = md - np.array([c * ms[0] - sn * ms[1], sn * ms[0] + c * ms[1]])
    warp = (c, -sn, float(tx), sn, c, float(ty))
    return warp, np.hypot(*(apply(warp, src) - dst).T)


def local_warp(src: np.ndarray, dst: np.ndarray, at: np.ndarray) -> Warp:
    """Similarity (rotation + shift + scale) fitted around `at`, every landmark weighted by its distance."""
    w = np.sqrt(np.exp(-np.hypot(*(src - at).T) ** 2 / (2 * SIGMA_PX ** 2)) + 1e-6)
    rows, rhs, weights = [], [], []
    for (x, y), (u, v), wi in zip(src, dst, w):
        rows += [[x, -y, 1, 0], [y, x, 0, 1]]
        rhs += [u, v]
        weights += [wi, wi]
    weights = np.array(weights)
    a, b, tx, ty = np.linalg.lstsq(np.array(rows) * weights[:, None], np.array(rhs) * weights, rcond=None)[0]
    return (float(a), float(-b), float(tx), float(b), float(a), float(ty))


def leave_one_out(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error (px) predicting every landmark from the others: global rigid vs local model."""
    rigid, local = [], []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        warp, _ = fit_rigid(src[keep], dst[keep])
        rigid.append(np.hypot(*(apply(warp, src[i:i + 1])[0] - dst[i])))
        local.append(np.hypot(*(apply(local_warp(src[keep], dst[keep], src[i]), src[i:i + 1])[0] - dst[i])))
    return np.array(rigid), np.array(local)


def spot_centres(rois: dict, size) -> dict[str, np.ndarray]:
    w, h = size
    return {s["id"]: np.array([sum(x for x, _ in s["points"]) / len(s["points"]) / 100 * w,
                               sum(y for _, y in s["points"]) / len(s["points"]) / 100 * h])
            for s in rois["spots"]}


def spot_warps(ref: Points, positions: Points, centres: dict[str, np.ndarray]) -> dict[str, Warp]:
    """Warp reference -> photo of every spot, from the landmark positions in the photo."""
    keys = [k for k in positions if k in ref]
    src = np.array([ref[k] for k in keys])
    dst = np.array([positions[k] for k in keys])
    return {sid: local_warp(src, dst, c) for sid, c in centres.items()}


# --- anchors (Label Studio) -----------------------------------------------------------------

def task_blob(task: dict) -> str | None:
    """Blob path of the photo (YYYY/MM/DD/parkwatch_<ts>Z.jpg): data.blob, or parsed from data.image."""
    data = task.get("data", {})
    if data.get("blob"):
        return data["blob"]
    m = PHOTO_RE.search(data.get("image", ""))
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(0)}" if m else None


def keypoints(task: dict) -> tuple[dict[str, list[float]], tuple[int, int]] | None:
    """{label: [x%, y%]} of the newest non-cancelled annotation and the image size (None if empty)."""
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    annotation = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    points, size = {}, None
    for r in annotation["result"]:
        if r.get("type") == "keypointlabels":
            points[r["value"]["keypointlabels"][0]] = [r["value"]["x"], r["value"]["y"]]
            size = (r["original_width"], r["original_height"])
    return (points, size) if points else None


def to_px(points: dict, size) -> Points:
    w, h = size
    return {k: np.array([x / 100 * w, y / 100 * h], float) for k, (x, y) in points.items()}


def drop_outliers(ref: Points, points: Points) -> tuple[Points, list[str]]:
    """Removes misclicks: the landmark the others predict worst, while it is further than OUTLIER_PX."""
    keys = [k for k in points if k in ref]
    dropped = []
    while len(keys) >= 4:
        _, err = leave_one_out(np.array([ref[k] for k in keys]), np.array([points[k] for k in keys]))
        worst = int(err.argmax())
        if err[worst] <= OUTLIER_PX:
            break
        dropped.append(keys.pop(worst))
    return {k: points[k] for k in keys}, dropped


def anchors_from_tasks(tasks: list[dict], reference_blob: str) -> tuple[Points, dict[str, Points], tuple, list[str]]:
    """(reference landmarks, {anchor blob: landmarks}, image size, warnings) from a Label Studio export."""
    marked, warnings = {}, []
    for task in tasks:
        blob, kp = task_blob(task), keypoints(task)
        if blob and kp:
            marked[blob] = kp
    if reference_blob not in marked:
        raise ValueError(f"the reference photo {reference_blob} has no landmarks in the landmarks project")
    ref_pct, size = marked.pop(reference_blob)
    ref = to_px(ref_pct, size)
    anchors = {reference_blob: ref}
    for blob, (pct, _) in sorted(marked.items()):
        points, dropped = drop_outliers(ref, to_px(pct, size))
        for k in dropped:
            warnings.append(f"{blob}: {k} does not fit the other landmarks (misclick?), ignored")
        if len(points) >= MIN_LANDMARKS:
            anchors[blob] = points
        else:
            warnings.append(f"{blob}: fewer than {MIN_LANDMARKS} usable landmarks, ignored")
    return ref, anchors, size, warnings


# --- tracking -------------------------------------------------------------------------------

def gray(data: bytes) -> np.ndarray:
    """Grey photo with equalised contrast (CLAHE): robust to lighting changes."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)


def find(image: np.ndarray, patch: np.ndarray, around: np.ndarray, radius: int) -> tuple[np.ndarray, float] | None:
    h, w = image.shape
    x0, y0 = int(around[0]) - PATCH - radius, int(around[1]) - PATCH - radius
    x1, y1 = int(around[0]) + PATCH + radius + 1, int(around[1]) + PATCH + radius + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    res = cv2.matchTemplate(image[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    return np.array([x0 + mx + PATCH, y0 + my + PATCH], float), float(score)


def movement(src: np.ndarray, dst: np.ndarray) -> tuple[Warp, np.ndarray]:
    """Rigid movement between two photos; only the shift when the matches are close together
    (a rotation estimated from a tight cluster swings the far landmarks)."""
    spread = np.hypot(*(src[:, None] - src[None]).transpose(2, 0, 1)).max()
    if spread >= SPREAD_ROTATION:
        return fit_rigid(src, dst)
    tx, ty = np.median(dst - src, axis=0)
    warp = (1.0, 0.0, float(tx), 0.0, 1.0, float(ty))
    return warp, np.hypot(*(apply(warp, src) - dst).T)


def coherent(previous: Points, found: Points) -> tuple[Points, Warp | None]:
    """Matches consistent with one rigid movement since the previous photo, and that movement."""
    keys = list(found)
    while len(keys) >= MIN_LANDMARKS:
        warp, residuals = movement(np.array([previous[k] for k in keys]), np.array([found[k] for k in keys]))
        worst = int(np.argmax(residuals))
        if residuals[worst] <= MAX_DISAGREE:
            return {k: found[k] for k in keys}, warp
        keys.pop(worst)
    return {}, None


def photo_time(blob: str) -> datetime:
    """UTC time of a photo from its name (parkwatch_YYYYMMDDTHHMMSSZ.jpg)."""
    return datetime.strptime(blob.rsplit("_", 1)[1][:15], "%Y%m%dT%H%M%S")


class Tracker:
    """Landmark positions photo by photo, in time order. `load(blob) -> bytes` reads a photo
    (used for the anchor templates)."""

    def __init__(self, ref: Points, anchors: dict[str, Points], load: Callable[[str], bytes]):
        self.ref, self.anchors, self.load = ref, anchors, load
        self._times = sorted((photo_time(b), b) for b in anchors)
        self._template: tuple[str, dict] | None = None
        self.positions: Points = {}
        self.missed = {k: 0 for k in ref}
        self.lost = False

    def nearest_anchor(self, blob: str) -> str:
        t = photo_time(blob)
        i = bisect.bisect_left(self._times, (t, ""))
        candidates = [self._times[j] for j in (i - 1, i) if 0 <= j < len(self._times)]
        return min(candidates, key=lambda c: abs(c[0] - t))[1]

    def start(self, positions: Points | None = None, missed: dict | None = None, lost: bool = False,
              near: str | None = None) -> None:
        """From given positions (previous run) or from the anchor nearest to `near`."""
        if positions is None:
            base = self.anchors[self.nearest_anchor(near)]
            offset = np.median(np.array([base[k] - self.ref[k] for k in base]), axis=0)
            positions = {k: base.get(k, self.ref[k] + offset) for k in self.ref}
        self.positions = {k: np.asarray(positions.get(k, self.ref[k]), float) for k in self.ref}
        self.missed = {k: (missed or {}).get(k, 0) for k in self.ref}
        self.lost = lost

    def patches(self, blob: str) -> dict[str, np.ndarray]:
        source = self.nearest_anchor(blob)
        if self._template is None or self._template[0] != source:
            image = gray(self.load(source))
            out = {}
            for k, (x, y) in self.anchors[source].items():
                patch = image[int(y) - PATCH:int(y) + PATCH + 1, int(x) - PATCH:int(x) + PATCH + 1]
                if k in self.ref and patch.shape == (2 * PATCH + 1, 2 * PATCH + 1):
                    out[k] = patch
            self._template = (source, out)
        return self._template[1]

    def step(self, blob: str, data: bytes) -> dict:
        """Tracks one photo. Returns {status: tracked|lost|anchor, found: [...], scores, before_anchor}."""
        image = gray(data)
        found, scores = {}, []
        for k, patch in self.patches(blob).items():
            radius = RADIUS_LOST if self.lost or self.missed[k] >= MISSED_WIDE else RADIUS
            hit = find(image, patch, self.positions[k], radius)
            if hit and hit[1] >= MIN_SCORE:
                found[k] = hit[0]
                scores.append(hit[1])
        found, motion = coherent(self.positions, found)
        if motion is not None:
            rotation = abs(math.degrees(math.atan2(motion[3], motion[0])))
            if rotation > MAX_STEP_ROTATION_DEG or (rotation > BIG_STEP_ROTATION_DEG and len(found) < BIG_STEP_MIN_LANDMARKS):
                found = {}
        if len(found) >= MIN_LANDMARKS:
            self.lost = False
            self.positions = {k: found[k] if k in found else apply(motion, self.positions[k][None])[0]
                              for k in self.positions}
            self.missed = {k: 0 if k in found else self.missed[k] + 1 for k in self.missed}
        else:
            self.lost = True
        result = {"status": "lost" if self.lost else "tracked", "found": sorted(found),
                  "score": float(np.mean(scores)) if scores else 0.0, "before_anchor": None}
        if blob in self.anchors:  # the clicks win; what the tracking said is kept for validation
            clicked = self.anchors[blob]
            result["before_anchor"] = None if self.lost else {k: v.copy() for k, v in self.positions.items()}
            common = [k for k in clicked if k in self.positions]
            motion, _ = fit_rigid(np.array([self.positions[k] for k in common]), np.array([clicked[k] for k in common]))
            self.positions = {k: clicked[k] if k in clicked else apply(motion, self.positions[k][None])[0]
                              for k in self.positions}
            self.missed = {k: 0 for k in self.missed}
            self.lost = False
            result["status"] = "anchor"
        return result
