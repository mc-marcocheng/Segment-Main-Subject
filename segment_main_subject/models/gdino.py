from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import \
    GroundingDinoObjectDetectionOutput

from .utils import DEVICE


@dataclass
class GDINODetectionOutput:
    detection_output: GroundingDinoObjectDetectionOutput
    input_ids: torch.Tensor


class GDINO:
    def build_model(self, ckpt_path: str | None = None, device=DEVICE):
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
    ) -> GroundingDinoObjectDetectionOutput:
        for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
        inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return GDINODetectionOutput(outputs, inputs.input_ids)

    def search_box_threshold(
            self,
            images_pil: list[Image.Image],
            outputs: GDINODetectionOutput,
            max_threshold: float = .5,
            text_threshold: float = .25,
        ) -> list[dict]:
        """
        Search for the best threshold for the detection.
        """
        image_target_sizes = [k.size[::-1] for k in images_pil]
        results = self.processor.post_process_grounded_object_detection(
            outputs.detection_output,
            outputs.input_ids,
            threshold=max_threshold,
            text_threshold=text_threshold,
            target_sizes=image_target_sizes,
        )
        for idx in range(len(images_pil)):
            if results[idx]["labels"]:
                continue
            l = 0.
            r = max_threshold
            while r - l > .05:
                m = (l + r) / 2
                result = self.processor.post_process_grounded_object_detection(
                    outputs.detection_output,
                    outputs.input_ids,
                    threshold=m,
                    text_threshold=text_threshold,
                    target_sizes=image_target_sizes,
                )[idx]
                if result["labels"]:
                    l = m
                else:
                    r = m
            results[idx] = self.processor.post_process_grounded_object_detection(
                outputs.detection_output,
                outputs.input_ids,
                threshold=l * .8,
                text_threshold=text_threshold,
                target_sizes=image_target_sizes,
            )[idx]

        return results
