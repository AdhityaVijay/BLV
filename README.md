# BLV Assistant — proactive hazard alerting + grounded conversational scene understanding

A local proof-of-concept assistive system for blind and low-vision users. Live webcam
video feeds two layers at once: a fast YOLOv8n detector that speaks proactive hazard
alerts, and an on-demand Ollama vision-LLM that answers questions about the scene and
holds follow-up conversation. An orchestrator arbitrates the single audio channel
between them.

Everything runs locally. No cloud APIs, no accounts, no data leaves the machine.

## Architecture

```
                    ┌──────────────┐
                    │   webcam     │
                    └──────┬───────┘
                           │ newest frame (reader thread; stale frames dropped)
              ┌────────────┴────────────┐
              │                         │
      ┌───────▼────────┐        ┌───────▼────────┐
      │  detector.py   │        │    vlm.py      │
      │  YOLOv8n, GPU  │        │  Ollama        │
      │  continuous    │        │  minicpm-v     │
      │                │        │  on demand     │
      └───────┬────────┘        └───────▲────────┘
              │                         │
              │   ┌─────────────────────┴──┐
              └──►│   scene_state.py       │
    tracked objs  │   GROUNDED BLACKBOARD  │  detector facts ──► every VLM prompt
    + geometry    │                        │  user intent    ◄── every VLM query
                  └─────────┬──────────────┘
                            │ severity, re-weighted by intent
                  ┌─────────▼──────────────┐
                  │   arbitration.py       │   allowed / interrupted /
                  │   turn-taking rules    │   queued / suppressed
                  └─────────┬──────────────┘
                            │
                  ┌─────────▼──────────────┐
                  │   tts.py (own thread)  │──► speech, interruptible
                  └────────────────────────┘
                            │
                  ┌─────────▼──────────────┐
                  │ logs/session_log.jsonl │  every event, timestamped
                  └────────────────────────┘
```

## What makes this different from a two-layer hybrid

The usual hybrid runs the two layers independently and has the orchestrator stop them
talking over each other. This one makes them **share state bidirectionally** through
`pipeline/scene_state.py`:

**Forward (detector → LLM).** The detector's measurements are injected into every LLM
prompt as a `DETECTOR FACTS` block, and the system prompt instructs the model to trust
them over the user's assertions. This targets sycophancy directly: when a user says
*"there's nothing in front of me, right?"* while a person stands 1.1 m ahead, the model
is holding contradicting evidence instead of only its own visual impression.

**Backward (LLM → detector).** The user's question is parsed into an intent, which
re-weights hazard salience. Saying *"can I walk forward?"* raises the severity of
obstacles in the walking path by one level. Asking *"where's my cup?"* makes the
proactive layer volunteer the cup when it comes into view — something a purely
geometric hazard layer would never do, because a cup is not a hazard.

The path can be switched off (`grounding.enabled: false`, or `--no-grounding`), which
is what makes it an **ablation rather than a claim**: `redteam_test.py --ablation` runs
the same adversarial probes under both conditions and prints the paired failure rates.

## Setup

Requires Python 3.10/3.11, an NVIDIA GPU (built against an RTX 4060, 8 GB), and Ollama.

```bash
python3.10 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
```

**GPU setup — do this before requirements.txt**, or you get CPU torch and the detector
will tell you so on startup:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

**Models:**

```bash
ollama serve                        # if not already running as a service
ollama pull minicpm-v               # ~5.5 GB, the reactive layer
ollama pull llama3.2-vision         # optional fallback, ~7.9 GB
# YOLOv8n weights download automatically on first detector run (~6 MB)
```

Both vision models fit in 8 GB VRAM individually. Running one alongside YOLOv8n is
fine; running both LLMs at once is not.

## Running

```bash
# Day 1 — proactive layer alone, no LLM
python -m pipeline.run_hazard_loop

# Day 2 — reactive layer alone, with latency numbers
python -m pipeline.run_reactive_test --images --repeat 3

# Day 3+ — the merged system (main entry point)
python -m pipeline.orchestrator
python -m pipeline.orchestrator --no-grounding    # ablation condition

# Day 6 — adversarial reliability testing (3 conditions, blind grading)
python -m pipeline.redteam_test --all-images --conditions all
python -m pipeline.redteam_test --grade logs/redteam_results.jsonl
python -m pipeline.redteam_test --summarize logs/redteam_results.jsonl

# Camera trouble? Find a working index/backend
python -m pipeline.probe_camera

# Day 7 — session analysis
python -m pipeline.analyze_logs --since-last-start
python -m pipeline.analyze_logs --mark-task-start "find the keys"

# Tests (no GPU, camera or Ollama needed)
pytest
```

**Orchestrator keys:** `SPACE` ask a typed question · `d` quick "what's in front of me?"
· `e` end conversation session · `q` quit.

## Design decisions worth knowing

**Spatial convention: left/right + distance, everywhere.** Not clock-face. A ~60° webcam
FOV compresses every visible object into 11–1 o'clock, so clock-face would carry almost
no information on this hardware. Clock-face is the better convention for orientation and
mobility generally, and would be worth revisiting with a wide-angle camera. The choice is
enforced in `prompts/system_prompts.yaml` and used identically by both layers.

**Distance is a pinhole estimate, not a box-height heuristic.** `distance = (known object
height × focal length) / box height in pixels`, with focal length derived from the
camera's FOV. The obvious heuristic ("big box = close") conflates object size with
distance — a bus across the room produces a taller box than a chair at arm's length, so
it announces the bus and stays quiet about the chair. See
`tests/test_geometry_severity.py::test_the_bug_the_naive_heuristic_had`.

**Speech runs on its own thread.** `pyttsx3.runAndWait()` blocks for the length of the
utterance. On the perception loop, that makes the system blind for the 2–3 seconds it
spends describing the last thing it saw — and OpenCV then hands back a buffered, stale
frame. The hazard it misses is exactly the one that arrived while it was talking, so
this is a safety property, not a performance tweak.

**Alert phrasing is deterministic, not LLM-generated.** The proactive path must be fast
and predictable; routing hazard wording through an LLM would add seconds and a
hallucination surface to the one output that must never be wrong. Templates live in the
prompt library so both layers still share one convention.

## Known limitations

Stated plainly — this is a feasibility prototype, not a validated system.

- **No formal evaluation with blind or low-vision users.** No pilot, no IRB, no
  recruitment. Every usability claim here comes from blindfolded self-testing by a
  sighted developer, which is a rehearsal, not evidence. Nothing in this repo should be
  read as demonstrating that the system helps its intended users.
- **The trigger mechanism is not accessible.** Keyboard input via a focused OpenCV
  window presumes a user who can see and focus that window. Speech-to-text is out of
  scope for this sprint, so the current build is a developer harness. This is the single
  largest gap between the prototype and anything usable.
- **COCO's 80 classes have no BLV-relevant hazards.** No stairs, curbs, potholes, glass
  doors, overhanging branches or drop-offs. The detector cannot warn about the hazards
  that most often injure blind pedestrians, and no threshold tuning changes that — it
  needs a fine-tuned model on a custom dataset.
- **Distance estimation assumes typical object heights.** A child, a seated person or a
  truncated bounding box all break the estimate. Accuracy is roughly ±20–30% at 1–4 m.
- **Single fixed camera, no depth sensor, no tracking across frames.** Objects have no
  identity between frames; the de-duplication key (class + position + distance band) is a
  proxy for identity and will merge two people standing in the same place.
- **The LLM layer is slow relative to walking.** Measure it yourself with
  `run_reactive_test.py --images --repeat 3`; expect seconds, not milliseconds. The
  reactive layer is for stationary questions, not live navigation.
- **Sycophancy is reduced, not solved.** Grounding gives the model contradicting
  evidence; it does not compel the model to use it. Quantify with `--ablation` rather
  than assuming.
- **Testing is single-session and informal.** Small image set, one environment, one
  tester.

## Repo layout

```
blv-assistant/
  pipeline/
    capture.py         webcam capture, threaded newest-frame reader
    detector.py        YOLOv8n + pinhole distance + severity rule
    scene_state.py     GROUNDED BLACKBOARD (the novel component)
    vlm.py             Ollama wrapper, multi-turn sessions, grounding injection
    arbitration.py     turn-taking rules (pure logic, unit tested)
    orchestrator.py    threaded entry point tying it all together
    tts.py             speech gate + interruptible speech thread
    run_hazard_loop.py       day 1 standalone proactive layer
    run_reactive_test.py     day 2 standalone reactive layer + latency
    redteam_test.py          day 6 adversarial probes with ablation
    analyze_logs.py          day 7 session analysis
  prompts/system_prompts.yaml    versioned prompt library
  logs/session_log.jsonl         every event, one JSON line each
  test_data/images/  clips/      your test media
  tests/                         72 tests, no hardware required
  docs/gap_and_approach.md       the gap, the approach, how to evaluate it
  config.yaml                    every tunable in one place
```
