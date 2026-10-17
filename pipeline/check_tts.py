"""Diagnose speech output. Run this when alerts are silent.

    python -m pipeline.check_tts

Tests each layer separately, because they fail for different reasons and the
fix is different in each case:

  1. pyttsx3 imports at all
  2. pywin32 / comtypes present (Windows SAPI5 is COM-based)
  3. engine constructs on the MAIN thread
  4. voices are available and audible
  5. engine constructs on a WORKER thread  <- the case the pipeline uses,
     and the one that fails on Windows when COM was never initialised there

Most Windows failures are step 5 passing on 3 but not on 5, which is why the
pipeline can look fine in a quick test and be mute in real use.
"""

from __future__ import annotations

import sys
import threading
import time


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_imports() -> bool:
    print("\n1. Imports")
    try:
        import pyttsx3  # noqa: F401
        _ok(f"pyttsx3 {getattr(pyttsx3, '__version__', 'unknown version')}")
    except Exception as e:
        _fail(f"pyttsx3 will not import: {type(e).__name__}: {e}")
        print("       fix:  pip install pyttsx3")
        return False

    if sys.platform == "win32":
        try:
            import pythoncom  # noqa: F401
            _ok("pythoncom (pywin32) available")
        except Exception as e:
            _fail(f"pythoncom missing: {e}")
            print("       Windows SAPI5 is COM-based; without this the engine")
            print("       cannot start on a worker thread.")
            print("       fix:  pip install pywin32")
            print("       then: python venv\\Scripts\\pywin32_postinstall.py -install")
            return False
        try:
            import comtypes  # noqa: F401
            _ok("comtypes available")
        except Exception as e:
            _fail(f"comtypes missing: {e}  ->  pip install comtypes")
            return False
    return True


def check_main_thread() -> bool:
    print("\n2. Engine on the main thread")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        _ok("engine constructed")
        voices = engine.getProperty("voices")
        if not voices:
            _fail("no voices installed — Windows: Settings > Time & language > "
                  "Speech > Manage voices")
            return False
        _ok(f"{len(voices)} voice(s): " + ", ".join(
            str(getattr(v, 'name', '?')) for v in voices[:3]))
        print("\n   Speaking a test phrase — you should HEAR this...")
        engine.say("Speech output is working. Testing one, two, three.")
        engine.runAndWait()
        _ok("spoke without error")
        try:
            engine.stop()
        except Exception:
            pass
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_worker_thread() -> bool:
    """The case that actually matters: the pipeline speaks from a thread."""
    print("\n3. Engine on a worker thread (what the pipeline does)")
    result: dict = {}

    def worker() -> None:
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
                result["com"] = True
            except Exception:
                result["com"] = False
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.say("Worker thread speech is working.")
            engine.runAndWait()
            result["ok"] = True
        except Exception as e:
            result["ok"] = False
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker, name="tts-check", daemon=True)
    t.start()
    t.join(timeout=25)

    if t.is_alive():
        _fail("worker thread hung (>25s) — the engine blocked on this thread")
        return False
    if result.get("com") is False and sys.platform == "win32":
        print("  [warn] CoInitialize() did not run")
    if result.get("ok"):
        _ok("spoke from a worker thread")
        return True
    _fail(result.get("error", "unknown failure"))
    return False


def check_pipeline_worker() -> bool:
    print("\n4. The pipeline's own TTSWorker")
    try:
        from pipeline.tts import TTSWorker
        w = TTSWorker(rate=185).start()
        print(f"  {w.health_line()}")
        if not w.is_healthy:
            _fail("TTSWorker reports an unhealthy engine")
            w.stop()
            return False
        w.say("The alert path is working.", priority=0, kind="alert")
        w.wait_until_idle(timeout=20)
        w.stop()
        _ok("TTSWorker spoke an alert")
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def main() -> None:
    print("=" * 66)
    print("SPEECH DIAGNOSTIC")
    print("=" * 66)
    print(f"platform: {sys.platform}   python: {sys.version.split()[0]}")
    print("\nTurn your volume up. Each stage below should be AUDIBLE.")

    stages = [
        ("imports", check_imports),
        ("main thread", check_main_thread),
        ("worker thread", check_worker_thread),
        ("pipeline worker", check_pipeline_worker),
    ]
    results = {}
    for name, fn in stages:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [FAIL] {name} raised {type(e).__name__}: {e}")
            results[name] = False
        if not results[name]:
            break
        time.sleep(0.3)

    print("\n" + "=" * 66)
    for name, _ in stages:
        status = {True: "pass", False: "FAIL"}.get(results.get(name), "skipped")
        print(f"  {name:<18}{status}")

    if all(results.get(n) for n, _ in stages):
        print("\nSpeech works everywhere it needs to.")
        print("If pipeline alerts are still silent, they are not being GENERATED.")
        print("Check the HUD's LAST ALERT panel: empty means no object reached")
        print("MEDIUM severity. Stand closer, fully in frame, or widen the bands")
        print("in config.yaml (detector.distance_bands_m).")
    elif results.get("main thread") and not results.get("worker thread"):
        print("\nDiagnosis: speech works on the main thread but not on a worker")
        print("thread. That is the COM-initialisation problem. Ensure pywin32 is")
        print("installed and its post-install script has run:")
        print("  pip install --force-reinstall pywin32")
        print("  python venv\\Scripts\\pywin32_postinstall.py -install")
    elif not results.get("main thread"):
        print("\nDiagnosis: the speech engine itself will not start. Try a known-")
        print("good pyttsx3 build, which resolves most SAPI5 driver problems:")
        print("  pip install pyttsx3==2.90")
        print("Also confirm Windows has voices installed and audio is not muted.")


if __name__ == "__main__":
    main()
