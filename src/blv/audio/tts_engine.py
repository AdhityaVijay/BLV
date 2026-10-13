"""Offline text-to-speech via pyttsx3 (no internet / API key required).

pyttsx3 is synchronous and blocks until speech finishes, which is fine
for a single-alert-at-a-time assistant like this one.
"""

from __future__ import annotations

import pyttsx3


class TTSEngine:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()
