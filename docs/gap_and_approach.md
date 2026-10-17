# Gap, approach, and how to evaluate the contribution

## The gap

Two findings motivate this prototype, and they point in the same direction.

First, vision-language models are markedly weaker at *proactive* assistance than at
*reactive* question answering. A model that describes a scene competently when asked
will often fail to volunteer that something dangerous is approaching. Reactive Q&A is
the task these models are trained and benchmarked on; unprompted hazard alerting is not.

Second, work probing live-video LLM assistance for blind users reports three reliability
failure modes: hallucination, generic non-personalised responses, and **sycophancy** —
the model accepting a user's false premise instead of correcting it. Sycophancy is
ordinarily a politeness problem. Here it is a safety problem: a user who says "the path
is clear, right?" and is agreed with may step forward into an obstacle.

The common architectural response is a hybrid: a fast detector for proactive alerts, a
vision-LLM for conversation. That addresses the first finding. It does not address the
second, because in the standard design the two layers are **independent**. They share a
camera and an audio channel, and the orchestrator's only job is to stop them speaking
over each other. The detector may hold a high-confidence measurement of the very object
the user is falsely denying — and no path exists to route that fact to the layer doing
the talking.

*Note on sourcing: the specific benchmark and paper framing this project cites should be
verified and cited properly before any write-up is submitted. The architectural argument
here does not depend on a particular benchmark number, but a citation that does not
check out will undermine everything around it.*

## The approach: a grounded scene-state blackboard

`pipeline/scene_state.py` makes the two layers share state in both directions.

**Forward path — the detector as a grounding oracle.** Every reactive query carries a
`DETECTOR FACTS` block built from the detector's output on the same frame: object class,
position, metric distance, whether it sits in the walking corridor, and detector
confidence. The system prompt (`grounded_contract` in the prompt library) instructs the
model to treat these measurements as more reliable than its own visual impression, and
to correct the user when their statement contradicts them.

Two design details matter more than they look:

- **Stale facts are withheld.** If the detector's last observation is older than
  `max_fact_age_s`, no facts are sent. Presenting two-second-old geometry as present-tense
  ground truth would trade sycophancy for confidently wrong grounding.
- **An empty detection is not an all-clear.** When the detector finds nothing, the block
  says so *and* states that the detector only recognises a fixed set of object types, so
  an empty list does not prove the space is safe. Without that sentence, grounding would
  actively manufacture false confidence on exactly the scenes (unusual hazards, unknown
  objects) where it is most dangerous.

**Backward path — conversation as a salience signal.** Each query is parsed into an
intent that re-weights hazard severity:

| Intent | Trigger | Effect |
|---|---|---|
| `navigating` | "can I walk forward", "is my path clear" | +1 severity for objects in the walking corridor |
| `inspecting` | "what's on the table" | −1 severity for non-critical alerts (never critical) |
| `searching` | "where's my cup" | the named object is announced when it enters view |

The `searching` case is the clearest demonstration that the coupling does real work: the
proactive layer volunteers a cup on a table, which no geometric hazard rule would ever
flag, because the conversation told it that object mattered. The `inspecting` damp is
capped so that a `CRITICAL` hazard is never quietened — safety is not negotiable against
conversational comfort.

## Arbitration

`pipeline/arbitration.py` holds the turn-taking rules as pure logic, separate from the
threading in the orchestrator. Given a severity and the current state of the audio
channel it returns `allowed`, `interrupted`, `queued`, or `suppressed`, each with a
reason, and each logged.

The rules: `CRITICAL` interrupts an in-flight reactive answer; `HIGH` is queued behind
it; `MEDIUM` and below are suppressed during a reactive turn, on the reasoning that by
the time an answer finishes, a chair three metres away is no longer a description of
where the user is standing. Queued alerts older than `queue_max_age_s` are dropped for
the same reason. Cooldown de-duplication applies to everything except `CRITICAL`:
repeating "stop, person ahead" is acceptable, missing it is not.

Separating these rules from the threads is what makes the day-3 claim testable.
`tests/test_arbitration.py` covers every severity × state cell deterministically, so a
regression appears in the test suite rather than during a live demo.

## How to evaluate the contribution

The contribution is falsifiable, and the harness to falsify it is in the repo.

**Ablation.** `python -m pipeline.redteam_test --image <img> --ablation` runs all ten
adversarial probes twice — grounding on, grounding off — against the same images and the
same prompts. Verdicts are assigned by hand (`--grade`), because auto-grading sycophancy
with the same class of model that produces it is circular. `--summarize` prints paired
failure rates per condition and per failure mode.

The hypothesis is that grounding reduces sycophancy substantially, since the model is
handed evidence that directly contradicts the false premise, while affecting
hallucination less, since grounding adds facts but does not stop the model from inventing
detail beyond them. **That is a prediction, not a result.** If grounding turns out not to
help, that is a publishable finding about the limits of prompt-level grounding, and it
should be reported as such rather than tuned until it agrees.

**Latency cost.** Grounding lengthens every prompt. `analyze_logs.py` reports median
latency for grounded versus ungrounded queries, so the reliability benefit can be
weighed against its cost in milliseconds rather than asserted for free.

**Arbitration behaviour.** The arbitration log is the evidence for how the two layers
actually interleaved — counts of queued, interrupted and suppressed decisions, collisions
with in-flight reactive turns, and how often user intent changed an alert's severity.

## What this does not establish

This is a feasibility prototype. It shows that the architecture runs in real time on one
consumer GPU and that the coupling is implementable and measurable. It does not show that
the system helps blind or low-vision users, because no blind or low-vision user has
tested it. The blindfold protocol in day 7 is a rehearsal by a sighted developer who
knows the room, the objects, and the system's failure modes — useful for catching
latency and phrasing problems, worthless as evidence of utility.

The honest framing for a write-up is: *an architecture and an evaluation method*, with
measured reliability numbers on a small internal probe set, and an explicit statement
that user validation is the next step and has not been done.
