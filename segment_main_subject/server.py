import base64
import os
from io import BytesIO

import litserve as ls
import numpy as np
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from .segment_main_subject import SegmentMainSubject
from .utils import create_rgba_masked_image, draw_image

PORT = os.environ.get("PORT", 8765)


class SegmentMainSubjectAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        """Initialize or load the SegmentMainSubject model."""
        self.model = SegmentMainSubject(sam_type="sam2.1_hiera_small", device=device)

    def decode_request(self, request) -> dict:
        """Decode the incoming request to extract parameters and image bytes.

        Assumes the request is sent as multipart/form-data with fields:
        - vllm_base_url: str (Optional)
            vLLM base URL. Usually http://localhost:8000/v1. Required if no text_prompt is provided.
        - sam_type: str
            Choices: [sam2.1_hiera_tiny, sam2.1_hiera_small, sam2.1_hiera_base_plus, sam2.1_hiera_large]
        - max_threshold: float (default: 0.5)
            Treat boxes above this threshold as valid.
        - text_threshold: float (default: 0.25)
        - text_prompt: str (Optional)
            The main subject to segment. If not provided, will use VLLM to determine the main subject.
        - visualize_annotations: bool (default: False)
            For visualization in Gradio app only.
        - image: UploadFile
        """
        # Extract form data
        vllm_base_url = request.get("vllm_base_url")
        sam_type = request.get("sam_type")
        max_threshold = float(request.get("max_threshold", 0.5))
        text_threshold = float(request.get("text_threshold", 0.25))
        text_prompt = request.get("text_prompt", "")
        visualize_annotations = request.get("visualize_annotations", False)

        # Extract image file
        image_file: UploadFile = request.get("image")
        if image_file is None:
            raise ValueError("No image file provided in the request.")

        image_bytes = image_file.file.read()

        return {
            "vllm_base_url": vllm_base_url,
            "sam_type": sam_type,
            "max_threshold": max_threshold,
            "text_threshold": text_threshold,
            "image_bytes": image_bytes,
            "text_prompt": text_prompt,
            "visualize_annotations": visualize_annotations,
        }

    def predict(self, inputs: dict) -> dict:
        """Perform prediction using the SegmentMainSubject model.

        Yields:
            dict: Contains the processed output image.
        """
        logger.info("Starting prediction with parameters:")
        logger.info(
            f"sam_type: {inputs['sam_type']}, \
max_threshold: {inputs['max_threshold']}, \
text_threshold: {inputs['text_threshold']}, \
text_prompt: {inputs['text_prompt']}"
        )

        if not inputs['text_prompt']:
            if inputs["vllm_base_url"] is None:
                logger.warning("No text prompt provided and no vLLM base URL set. Cannot determine main subject.")
                raise ValueError("No text prompt provided and no vLLM base URL set.")
            elif inputs["vllm_base_url"] != self.model.vllm_base_url:
                logger.info(f"Updating vLLM base url to {inputs['vllm_base_url']}")
                self.model.vllm.build_model(base_url=inputs['vllm_base_url'])
                self.model.vllm_base_url = inputs['vllm_base_url']
        if inputs["sam_type"] != self.model.sam_type:
            logger.info(f"Updating SAM model type to {inputs['sam_type']}")
            self.model.sam.build_model(inputs["sam_type"])
            self.model.sam_type = inputs["sam_type"]

        try:
            image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

        if not inputs['text_prompt']:
            # Use VLLM to get the main subject, then predict from text
            results, texts_prompt = self.model.predict(images_pil=[image_pil])
            inputs["text_prompt"] = texts_prompt[0]
        else:
            # Predict from provided text prompt
            results = self.model.predict_from_texts(
                images_pil=[image_pil],
                texts_prompt=[inputs["text_prompt"]],
                max_threshold=inputs["max_threshold"],
                text_threshold=inputs["text_threshold"],
            )
        results = results[0]

        if not len(results["masks"]):
            logger.warning("No masks detected. Returning original image.")
            return {"output_image": image_pil}

        # Draw results on the image
        image_array = np.asarray(image_pil)
        output_image = create_rgba_masked_image(image_array, results["masks"])
        output_image = Image.fromarray(np.uint8(output_image), mode="RGBA")
        outputs = {"output_image": output_image, "text_prompt": inputs["text_prompt"]}

        # Gradio app visualization
        if inputs["visualize_annotations"]:
            annotation_image = draw_image(
                image_array,
                results["masks"],
                results["boxes"],
                results["scores"],
                results["labels"],
            )
            annotation_image = Image.fromarray(np.uint8(annotation_image)).convert("RGB")
            outputs["annotation_image"] = annotation_image

        return outputs

    def encode_response(self, output: dict) -> JSONResponse:
        """Encode the prediction result into an HTTP response.

        Returns:
            Response: Contains the processed image in PNG format, the predicted main subject in text,
            and the annotation image if visualization is enabled.
        """
        try:
            image = output["output_image"]
            text_prompt = output["text_prompt"]

            # Convert image to base64 string
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Create response JSON
            response_json = {
                "image": image_base64,
                "text_prompt": text_prompt,
            }

            if "annotation_image" in output:
                annotation_image = output["annotation_image"]
                # Convert image to base64 string
                buffer = BytesIO()
                annotation_image.save(buffer, format="PNG")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                response_json["annotation_image"] = image_base64

            return JSONResponse(content=response_json)
        except StopIteration:
            raise ValueError("No output generated by the prediction.")


lit_api = SegmentMainSubjectAPI()
server = ls.LitServer(lit_api)


if __name__ == "__main__":
    logger.info(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
