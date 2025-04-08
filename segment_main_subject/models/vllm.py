import base64
import json
import re

import cv2
import numpy as np
from openai import OpenAI
from PIL.Image import Image as ImageType


def extract_json_str(response: str) -> str:
    """
    Extracts the JSON string from the response.
    """
    # Find the JSON string in the response
    match = re.search(r'```json\s*(.+)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response


class VLLMMainSubject:
    def __init__(self):
        self.client_inited = False

    def build_model(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(base_url=base_url, api_key="DUMMY")
        models = self.client.models.list()
        self.model_id = models.data[0].id
        self.client_inited = True

    def get_main_subject(self, image: str | np.ndarray | ImageType, return_metadata: bool = False):
        """
        Get the main subject of the image using VLLM.
        """
        if not isinstance(image, str):
            image = np.asarray(image)
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_base64}"
        else:
            image_url = image

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that only speaks in JSON."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """1. Describe this image in short sentence.
2. What is the main subject in this image? Answer only the name of the main subject.

Answer in the following format:
```json
{
    "description": <short image description>,
    "main_subject": <main subject name>
}
```"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=512,
            temperature=0.,
        )

        generated_text = completion.choices[0].message.content
        generated_text = extract_json_str(generated_text)
        try:
            subject_name = json.loads(generated_text)["main_subject"]
        except:
            subject_name = generated_text

        if return_metadata:
            return subject_name, generated_text
        return subject_name

    def simplify_word(self, word: str, return_metadata: bool = False):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that only speaks in JSON. Your job is to simply an uncommon word given by user. If the word is common, return the same word without changes.

Example: Simplify the word "tench".
Your response:
```json
{
    "is_uncommon": 1,
    "simple_word": "fish"
}
```

Example: Simplify the word "fish".
Your response:
```json
{
    "is_uncommon": 0,
    "simple_word": "fish"
}
```

Example: Simplify the word "water snake".
Your response:
```json
{
    "is_uncommon": 1,
    "simple_word": "snake"
}
```
"""
            },
            {
                "role": "user",
                "content": 'Simplify the word "' + word + '".' + """

Answer in the following format:
```json
{
    "is_uncommon": <0 or 1>,
    "simple_word": <simple word>
}
```"""
            },
        ]
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=64,
            temperature=0.,
        )

        generated_text = completion.choices[0].message.content
        generated_text = extract_json_str(generated_text)
        try:
            simple_word = json.loads(generated_text)["simple_word"]
        except:
            simple_word = generated_text

        if return_metadata:
            return simple_word, generated_text
        return simple_word
