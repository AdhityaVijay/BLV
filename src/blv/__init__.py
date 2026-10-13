"""BLV — a desktop prototype assistant for blind/low-vision users.

Package layout:
    capture/     camera frame acquisition
    detection/   real-time obstacle/object detection (YOLOv8)
    description/ on-demand scene description (BLIP captioning)
    audio/       text-to-speech + alert prioritization/cooldown
    utils/       shared helpers (geometry, logging)
"""

__version__ = "0.1.0"
