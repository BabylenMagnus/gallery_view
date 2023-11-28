import pandas as pd

import os
import shutil
import glob

import json

import numpy as np
from PIL import Image
import io

import requests

from utils import A111_url, to_b64

import base64


INPUT_PATH = "GEN_IMG"
OUT_PATH = "UPSCALED_IMGS"
model_name = "4xUltrasharp_4xUltrasharpV10"

os.makedirs(OUT_PATH, exist_ok=True)

for i in os.listdir(INPUT_PATH):
    img = Image.open(os.path.join(INPUT_PATH, i))
    payload = {
        "resize_mode": 1,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 816,
        "upscaling_resize_h": 1088,
        "upscaler_1": model_name,
        "image": to_b64(np.array(img))
    }

    response = requests.post(f'{A111_url}/sdapi/v1/extra-single-image', json=payload)
    Image.open(io.BytesIO(base64.b64decode(response.json()['image']))).save(os.path.join(OUT_PATH, i))


