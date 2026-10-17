"""Day 7: parse session_log.jsonl into the numbers the write-up needs.

Reports reactive-query latency (median/p90/max), proactive alert counts and
severity mix, the arbitration breakdown (how often the two layers actually
collided and what the arbiter did), intent/grounding usage, and dogfooding
task durations if task markers were logged.

Run:
    python -m pipeline.analyze_logs
    python -m pipeline.analyze_logs --log logs/session_log.jsonl --since-last-start
    python -m pipeline.analyze_logs --mark-task-start "find the keys"
    python -m pipeline.analyze_logs --mark-task-end "find the keys" --completed yes
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.config_util import load_config, resolve_path
from pipeline.logging_util import EventLogger


def load_events(path: Path, since_last_start: bool = False) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No log at {path}. Run the pipeline first.")
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if since_last_start:
        last = None
        for i, e in enumerate(events):
            if e.get("type") == "system" and e.get("event") == "start":
                last = i
        if last is not None:
            events = events[last:]
    return events


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def report(events: list[dict]) -> None:
    queries = [e for e in events if e.get("type") == "query"]
    alerts = [e for e in events if e.get("type") == "alert"]
    detections = [e for e in events if e.get("type") == "detection"]
    arb = [e for e in events if e.get("type") == "arbitration"]
    intents = [e for e in events if e.get("type") == "intent"]
    tasks = [e for e in events if e.get("type") == "task"]

    print("=" * 66)
    print("SESSION ANALYSIS")
    print("=" * 66)
    print(f"events={len(events)}  detections={len(detections)}  alerts={len(alerts)}  "
          f"queries={len(queries)}  arbitrations={len(arb)}")

    # -- reactive latency --------------------------------------------------
    print("\n-- Reactive query latency (the number people ask about) --")
    ok = [q for q in queries if not q.get("error") and q.get("latency_ms")]
    if ok:
        lat = [float(q["latency_ms"]) for q in ok]
        print(f"  n={len(lat)}  median={statistics.median(lat):.0f} ms  "
              f"p90={pct(lat, 90):.0f} ms  max={max(lat):.0f} ms")
        if len(lat) > 1:
            print(f"  excluding first call (model load): median="
                  f"{statistics.median(lat[1:]):.0f} ms")
        grounded = [float(q["latency_ms"]) for q in ok if q.get("grounded")]
        plain = [float(q["latency_ms"]) for q in ok if not q.get("grounded")]
        if grounded and plain:
            print(f"  grounded median={statistics.median(grounded):.0f} ms "
                  f"(n={len(grounded)})  vs ungrounded={statistics.median(plain):.0f} ms "
                  f"(n={len(plain)})")
            print("  ^ the cost of the grounding block, in milliseconds")
    else:
        print("  no successful queries logged")
    errs = [q for q in queries if q.get("error")]
    if errs:
        print(f"  ERRORS: {len(errs)}")
        for e in errs[:3]:
            print(f"    - {str(e.get('error'))[:90]}")

    # -- detector ----------------------------------------------------------
    print("\n-- Detector --")
    if detections:
        dl = [float(d["frame_latency_ms"]) for d in detections if d.get("frame_latency_ms")]
        if dl:
            print(f"  inference median={statistics.median(dl):.1f} ms  max={max(dl):.1f} ms")
        classes = Counter(d.get("object_class") for d in detections)
        print(f"  classes seen: " + ", ".join(f"{c}({n})" for c, n in classes.most_common(8)))
    else:
        print("  no detections logged")

    # -- proactive alerts --------------------------------------------------
    print("\n-- Proactive alerts --")
    if alerts:
        sev = Counter(a.get("severity") for a in alerts)
        print(f"  spoken={len(alerts)}  by severity: "
              + ", ".join(f"{s}={n}" for s, n in sev.most_common()))
        intent_driven = [a for a in alerts if a.get("decision") == "intent_driven"]
        if intent_driven:
            print(f"  intent-driven target announcements: {len(intent_driven)} "
                  f"(backward path fired)")
    else:
        print("  none spoken")

    # -- arbitration (the day-3 evidence) ----------------------------------
    print("\n-- Arbitration (how the two layers interleaved) --")
    if arb:
        dec = Counter(a.get("decision") for a in arb)
        total = sum(dec.values())
        for d, n in dec.most_common():
            print(f"  {d:<14}{n:>6}  {n / total:>6.0%}")
        collisions = [a for a in arb if a.get("decision") in ("queued", "interrupted", "suppressed")
                      and "reactive" in str(a.get("reason", ""))]
        print(f"  collisions with an in-flight reactive turn: {len(collisions)}")
        interrupts = [a for a in arb if a.get("decision") == "interrupted"]
        if interrupts:
            print(f"  safety interrupts: {len(interrupts)}")
            for i in interrupts[:3]:
                print(f"    - {i.get('effective_severity')} {i.get('object_class')} "
                      f"@{i.get('distance_m')}m: {i.get('reason')}")
        escalated = [a for a in arb
                     if a.get("base_severity") and a.get("effective_severity")
                     and a["base_severity"] != a["effective_severity"]]
        if escalated:
            print(f"  severity changed by user intent: {len(escalated)} "
                  f"(e.g. {escalated[0]['base_severity']}->{escalated[0]['effective_severity']} "
                  f"under intent '{escalated[0].get('intent_mode')}')")
    else:
        print("  no arbitration decisions logged (did any alert fire during a query?)")

    # -- grounding / intent ------------------------------------------------
    print("\n-- Grounding & intent (novel path) --")
    if intents:
        modes = Counter(i.get("mode") for i in intents)
        print(f"  intents parsed: " + ", ".join(f"{m}={n}" for m, n in modes.most_common()))
    grounded_q = sum(1 for q in queries if q.get("grounded"))
    if queries:
        print(f"  queries carrying detector facts: {grounded_q}/{len(queries)}")

    # -- dogfooding tasks --------------------------------------------------
    print("\n-- Dogfooding tasks --")
    if tasks:
        starts: dict[str, float] = {}
        durations: list[tuple[str, float, str]] = []
        for t in tasks:
            name = t.get("task_name", "?")
            if t.get("event") == "start":
                starts[name] = float(t.get("t_mono", 0))
            elif t.get("event") == "end" and name in starts:
                durations.append((name, float(t.get("t_mono", 0)) - starts.pop(name),
                                  str(t.get("completed", "?"))))
        if durations:
            print(f"{'task':<38}{'seconds':>9}  completed")
            for name, dur, done in durations:
                print(f"  {name:<36}{dur:>9.1f}  {done}")
        for unfinished in starts:
            print(f"  {unfinished:<36}{'--':>9}  never marked complete")
    else:
        print("  no task markers logged (use --mark-task-start / --mark-task-end)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze session_log.jsonl (day 7)")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--since-last-start", action="store_true",
                        help="only analyze events after the most recent pipeline start")
    parser.add_argument("--mark-task-start", type=str, help="log a dogfooding task start marker")
    parser.add_argument("--mark-task-end", type=str, help="log a dogfooding task end marker")
    parser.add_argument("--completed", type=str, default="unknown",
                        help="with --mark-task-end: yes|no|partial")
    args = parser.parse_args()

    cfg = load_config()
    log_path = resolve_path(args.log) if args.log else resolve_path(cfg["logging"]["session_log"])

    if args.mark_task_start or args.mark_task_end:
        logger = EventLogger(log_path)
        if args.mark_task_start:
            logger.log("task", event="start", task_name=args.mark_task_start)
            print(f"[task] start: {args.mark_task_start}")
        if args.mark_task_end:
            logger.log("task", event="end", task_name=args.mark_task_end,
                       completed=args.completed)
            print(f"[task] end: {args.mark_task_end} (completed={args.completed})")
        logger.close()
        return

    report(load_events(log_path, since_last_start=args.since_last_start))


if __name__ == "__main__":
    main()
