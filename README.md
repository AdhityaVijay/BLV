# BLV — Blind/Low-Vision Desktop Assistant (prototype)

A laptop-webcam prototype that:
1. **Watches continuously** — runs YOLOv8 obstacle detection on the webcam feed and speaks
   short alerts like "person close on your left".
2. **Describes on demand** — press `d` and a BLIP captioning model describes the current
   scene in a sentence, spoken aloud.

This is a *desktop prototype*, not the final deployment target. See `docs/architecture.md`
for how to evolve it toward phone/edge deployment later.

## Folder structure

```
blv/
├── config/
│   └── config.yaml          # all tunables live here (camera, thresholds, TTS, keys)
├── src/blv/
│   ├── capture/camera.py    # webcam wrapper (cv2.VideoCapture)
│   ├── detection/
│   │   └── object_detector.py   # YOLOv8 obstacle detection
│   ├── description/
│   │   └── scene_describer.py   # BLIP scene captioning (on keypress)
│   ├── audio/
│   │   ├── tts_engine.py        # pyttsx3 offline text-to-speech
│   │   └── alert_manager.py     # cooldown/priority logic so alerts don't spam
│   ├── utils/
│   │   ├── geometry.py          # bbox -> "close on your left" phrasing
│   │   └── logger.py
│   └── main.py               # orchestration loop (entry point)
├── tests/                    # pure-logic unit tests (no camera/model needed)
├── scripts/setup_venv.sh     # one-shot venv + install
├── models/                   # downloaded model weights (gitignored)
├── data/samples/             # sample images/video for offline testing
├── docs/architecture.md
├── requirements.txt
└── pyproject.toml
```

**Why this layout:** each concern (capture / detection / description / audio) is an
independently testable module with a narrow interface, so you can swap YOLOv8n for a
fine-tuned model, or BLIP for a cloud VLM, without touching `main.py`'s control flow.
`config.yaml` keeps every tunable (thresholds, cooldowns, camera index) out of code so
you can experiment without redeploying.

## Setup

Requires Python 3.10 or 3.11 (3.12+ can have wheel-availability issues with some
`torch`/`ultralytics` combos as of writing — check before deviating).

```bash
cd blv
bash scripts/setup_venv.sh
source venv/bin/activate
```

This creates `venv/`, installs `requirements.txt`, and installs the `blv` package itself
in editable mode (`pip install -e .`) so `import blv...` works from anywhere.

If you have an NVIDIA GPU and want CUDA acceleration instead of the CPU torch build,
install torch separately **before** the rest of requirements.txt, following the command
for your CUDA version at https://pytorch.org/get-started/locally/, then re-run
`pip install -r requirements.txt` (pip will skip torch since it's already satisfied).

## Run

```bash
python -m blv.main
```

- Press **`d`** to hear a description of the current scene.
- Press **`q`** to quit.

First run will download the YOLOv8n weights (~6MB) and the BLIP model (~1GB) —
this needs internet access once; after that both are cached locally.

## Test

```bash
pytest
```

Only `geometry.py` and `alert_manager.py` are unit tested here — they're pure logic.
`camera.py`, `object_detector.py`, and `scene_describer.py` need real hardware/models,
so they're best checked by running the app itself; if you want automated coverage for
them later, mock `cv2.VideoCapture` and the model `.predict()`/`.generate()` calls.

## Next steps (not built yet)

- Distance estimation is a height-ratio heuristic (`utils/geometry.py`). Swap in a
  monocular depth model (e.g. MiDaS via `torch.hub`) for real distances.
- No mobile deployment yet — see `docs/architecture.md` for the path to phone/edge.
- No fine-tuning — YOLOv8n here is the stock COCO-pretrained checkpoint (80 general
  object classes only; doesn't know "staircase" or "curb" out of the box).
