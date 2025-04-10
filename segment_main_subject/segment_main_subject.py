import numpy as np
from PIL import Image

from .models.gdino import GDINO
from .models.sam import SAM
from .models.utils import DEVICE
from .models.vllm import VLLMMainSubject


class SegmentMainSubject:
    def __init__(self, vllm_base_url: str | None = None, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.vllm = VLLMMainSubject()
        self.vllm_base_url = vllm_base_url
        if vllm_base_url:
            self.vllm.build_model(base_url=vllm_base_url)
        self.gdino = GDINO()
        self.gdino.build_model(device=device)
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path, device=device)

    def predict(
            self,
            images_pil: list[Image.Image],
            max_threshold: float = .5,
            text_threshold: float = .25,
        ) -> tuple[list[dict], list[str]]:
        texts_prompt = [self.vllm.get_main_subject(image) for image in images_pil]
        return self.predict_from_texts(images_pil, texts_prompt, max_threshold=max_threshold, text_threshold=text_threshold), texts_prompt

    def predict_from_texts(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        max_threshold: float = .5,
        text_threshold: float = .25,
    ) -> list[dict]:
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_outputs = self.gdino.predict(images_pil, texts_prompt)
        gdino_results = self.gdino.search_box_threshold(images_pil, gdino_outputs, max_threshold=max_threshold, text_threshold=text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
        return all_results
