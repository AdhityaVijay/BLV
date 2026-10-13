"""Pure-logic helpers to turn a detector's bounding box into something
speakable: a rough left/center/right position and a near/far distance
estimate. No model or camera dependency — this is why it's cheap to
unit test (see tests/test_geometry.py).
"""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2


def horizontal_position(box: BoundingBox, frame_width: int) -> str:
    """Classify a box as 'left', 'center', or 'right' of the frame."""
    ratio = box.center_x / frame_width
    if ratio < 0.35:
        return "left"
    if ratio > 0.65:
        return "right"
    return "center"


def relative_distance(box: BoundingBox, frame_height: int) -> str:
    """Very rough distance heuristic: a taller bounding box (relative to
    frame height) means the object is closer to the camera. This is NOT
    a real depth estimate — swap in a monocular depth model (e.g. MiDaS)
    if you need real distances.
    """
    ratio = box.height / frame_height
    if ratio > 0.6:
        return "very close"
    if ratio > 0.35:
        return "close"
    if ratio > 0.15:
        return "nearby"
    return "far"


def describe_position(box: BoundingBox, frame_width: int, frame_height: int) -> str:
    """Combine horizontal position + distance into a short spoken phrase,
    e.g. 'close on your left'.
    """
    pos = horizontal_position(box, frame_width)
    dist = relative_distance(box, frame_height)
    if pos == "center":
        return f"{dist} ahead"
    return f"{dist} on your {pos}"
