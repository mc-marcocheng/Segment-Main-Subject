# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import base64
import os
from io import BytesIO

import dotenv
import requests
from loguru import logger
from PIL import Image

dotenv.load_dotenv()

PORT = os.getenv("PORT", "8765")

def inference(vllm_base_url, sam_type, max_threshold, text_threshold, image, text_prompt=None, visualize_annotations=False):
    """Makes a request to the /predict LitServe endpoint."""
    url = f"http://127.0.0.1:{PORT}/predict"  # Adjust port if needed

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
            "visualize_annotations": visualize_annotations,
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
            output_image = Image.open(BytesIO(image_data)).convert("RGB")
            text_prompt = response_json["text_prompt"]
            if "annotation_image" in response_json:
                annotation_image_data = base64.b64decode(response_json["annotation_image"])
                annotation_image = Image.open(BytesIO(annotation_image_data)).convert("RGB")
                return output_image, annotation_image, text_prompt
            return output_image, text_prompt
        except Exception as e:
            logger.error(f"Failed to process response image: {e}")
            return None
    else:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for SegmentMainSubject API')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--vllm_base_url', type=str, default='http://localhost:8000/v1', help='vLLM base URL')
    parser.add_argument('--sam_type', type=str, default='sam2.1_hiera_small', help='SAM model type')
    parser.add_argument('--max_threshold', type=float, default=0.5, help='Max threshold value')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold value')
    parser.add_argument('--text_prompt', type=str, default=None, help='Text prompt (optional)')
    parser.add_argument('--visualize_annotations', action='store_true', help='Visualize annotations in the response')

    args = parser.parse_args()

    result = inference(
        vllm_base_url=args.vllm_base_url,
        sam_type=args.sam_type,
        max_threshold=args.max_threshold,
        text_threshold=args.text_threshold,
        image=args.image,
        text_prompt=args.text_prompt,
        visualize_annotations=args.visualize_annotations,
    )

    if result:
        if args.visualize_annotations:
            output_image, annotation_image, text_prompt = result
            output_image.show(title="Output Image")
            annotation_image.show(title="Output Annotation")
        else:
            output_image, text_prompt = result
            output_image.show(title="Output Image")
        print(f"Text Prompt: {text_prompt}")
