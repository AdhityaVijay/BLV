"""On-demand scene description (VQA/captioning) using a local BLIP model
via Hugging Face `transformers`. Triggered by keypress rather than run
every frame — captioning models are much heavier than YOLOv8n and don't
need to run in real time for this use case.

Swap-out point: if you later want higher-quality descriptions and have
network access, replace `describe()`'s body with a call to a cloud VLM
(e.g. GPT-4o-mini or Gemini) behind the same interface — main.py doesn't
need to change.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class SceneDescriber:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    def describe(self, frame_bgr: np.ndarray, max_new_tokens: int = 40) -> str:
        # OpenCV frames are BGR; PIL/BLIP expect RGB.
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)
