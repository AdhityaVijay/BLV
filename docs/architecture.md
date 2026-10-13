# Architecture notes

## Current (Phase 1 — this scaffold)

```
Webcam --> Camera --> ObjectDetector (YOLOv8n, every Nth frame)
                   \-> SceneDescriber (BLIP, on keypress 'd')
                          |
                          v
                    AlertManager (cooldown/priority)
                          |
                          v
                     TTSEngine (pyttsx3, offline)
```

Everything runs in a single Python process on a laptop. This is intentionally the
simplest possible loop so you can validate the core idea (does obstacle-alert + on-demand
description actually feel useful?) before investing in mobile/edge packaging.

## Phase 2 — phone + cloud backend (if the prototype validates)

```
Phone camera --(frames over HTTP/WebSocket)--> FastAPI backend --> same
  detection/description modules --> JSON response --> phone TTS
```

`src/blv/detection` and `src/blv/description` don't need to change — only `main.py`'s
control loop gets replaced by a FastAPI endpoint that accepts a frame and returns
detections/captions. This is why those modules take a raw frame (`np.ndarray`) in and
return plain dataclasses/strings out, with no camera or audio code inside them.

## Phase 3 — fully on-device (edge)

- Replace `ultralytics` YOLOv8n with a quantized export (ONNX/TFLite) for mobile GPUs/NPUs.
- Replace BLIP-base with a distilled/quantized captioning model, or drop scene description
  in favor of obstacle-only alerts if latency/battery is too tight.
- Replace `pyttsx3` with the platform's native TTS (Android `TextToSpeech`, iOS `AVSpeechSynthesizer`).

## Known simplifications in this scaffold

- **Distance** is a bounding-box-height heuristic, not real depth. Fine for "something is
  very close" but not for precise navigation. MiDaS (monocular depth, `torch.hub`) is the
  natural upgrade if you stay on a single RGB camera.
- **Obstacle classes** are whatever stock COCO gives you (80 classes: person, car, chair,
  etc.) — no "curb", "pothole", or "staircase" detection without fine-tuning on a custom
  dataset.
- **Single-threaded**: detection, description, and TTS all block the main loop. Fine for a
  prototype; for smoother real-time behavior later, move TTS (and ideally detection) onto
  background threads/processes with a small queue.
