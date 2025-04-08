import base64
import os
from io import BytesIO

import dotenv
import gradio as gr
import requests
from loguru import logger
from PIL import Image

dotenv.load_dotenv()

from segment_main_subject import SAM_MODELS
from segment_main_subject.server import PORT, server


def inference(vllm_base_url, sam_type, max_threshold, text_threshold, image, text_prompt):
    """Gradio function that makes a request to the /predict LitServe endpoint."""
    url = f"http://localhost:{PORT}/predict"  # Adjust port if needed

    # Prepare the multipart form data
    with open(image, "rb") as img_file:
        files = {
            "image": img_file,
        }
        data = {
            "vllm_base_url": vllm_base_url,
            "sam_type": sam_type,
            "max_threshold": str(max_threshold),
            "text_threshold": str(text_threshold),
            "text_prompt": text_prompt,
            "visualize_annotations": True,
        }

        try:
            response = requests.post(url, files=files, data=data)
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    if response.status_code == 200:
        response_json = response.json()
        try:
            image_data = base64.b64decode(response_json["image"])
            annotation_image_data = base64.b64decode(response_json["annotation_image"])
            output_image = Image.open(BytesIO(image_data)).convert("RGB")
            annotation_image = Image.open(BytesIO(annotation_image_data)).convert("RGB")
            text_prompt = response_json["text_prompt"]
            return output_image, annotation_image, text_prompt
        except Exception as e:
            logger.error(f"Failed to process response image: {e}")
            return None
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        return None

examples = [
    [
        os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
        "kiwi. watermelon. blueberry.",
    ],
    [
        os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
        "wheel.",
    ],
    [
        os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
        "",
    ],
]

with gr.Blocks(title="Segment Main Subject", analytics_enabled=False) as blocks:
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Input Image")
                text_prompt = gr.Textbox(lines=1, label="Text Prompt (Optional)")
                submit_btn = gr.Button("Run Prediction", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image")
                output_annotation = gr.Image(type="pil", label="Output Annotation")
        gr.Examples(
            examples=examples,
            inputs=[image_input, text_prompt],
        )

    with gr.Tab("Settings"):
        vllm_base_url_textbox = gr.Textbox("http://localhost:8000/v1", label="vLLM base URL")
        sam_model_choices = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM Model", value="sam2.1_hiera_small")
        max_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Max Threshold")
        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text Threshold")

    submit_btn.click(
        fn=inference,
        inputs=[vllm_base_url_textbox, sam_model_choices, max_threshold, text_threshold, image_input, text_prompt],
        outputs=[output_image, output_annotation, text_prompt],
    )

server.app = gr.mount_gradio_app(server.app, blocks, path="/gradio")

if __name__ == "__main__":
    logger.info(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
