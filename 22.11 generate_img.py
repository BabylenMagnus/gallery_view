import random
import time

import os

import numpy as np
from PIL import Image
import json


from tqdm import tqdm
from datetime import datetime
from uuid import uuid4

from utils import MODELS, generate_image_txt2img, generate_image_upgrade


OUT_PATH = "RESULT"
PROMPTS_PATH = "/home/jjjj/Documents/gallery_view/IMGS_22.11/prompts.json"
INPUT_PATH = r"/home/jjjj/Documents/gallery_view/IMGS_22.11/without"

with open(PROMPTS_PATH, "r") as t:
    data = json.load(t)

models_ = []
for i in MODELS:
    v = "sdxl_vae.safetensors"
    if i == "CutifiedAnimeCharacterDesign/cutifiedanimecharact_sdxlV10.safetensors":
        v = "XL_VAE_E7.safetensors"
    models_.append((i, v))

random.shuffle(models_)

neg_prompt = "NSFW, bad anatomy, bad hands, missing fingers, text"

sampler = ["DPM++ 3M SDE Exponential", "DPM++ 3M SDE Karras", "Euler a"]
steps = [25, 30, 40, 60, 70]
cfg_scale = [4, 6, 7.5]
denoising_strength = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
batch_size = 8

X = os.listdir(INPUT_PATH)
random.shuffle(X)
for i in tqdm(X):
    if i[:-4] not in data:
        continue

    out_path = os.path.join(OUT_PATH, i[:-4])

    if os.path.exists(out_path) and len(os.listdir(out_path)) > 50:
        continue

    os.makedirs(out_path)

    img = Image.open(os.path.join(INPUT_PATH, i))
    random.shuffle(models_)

    try:
        for model_name, vae in models_[:3]:
            for prompt in [i[:-4], data[i[:-4]]]:
                s = random.choice(steps)
                cfg__ = random.choice(cfg_scale)
                den__ = random.choice(denoising_strength)

                imgs = generate_image_txt2img(
                    model_name, vae, prompt, neg_prompt, random.choice(sampler),
                    s, cfg__, den__, batch_size
                )
                for im in imgs:
                    im.save(
                        os.path.join(
                            out_path, model_name[:5] + "_" + str(s) + "_" + str(cfg__) + "_" + str(den__) + "_"
                                      + datetime.now().strftime("%d %H %M") + f"_{str(uuid4())[:4]}.png"
                        )
                    )

                imgs = generate_image_upgrade(
                    np.array(img), model_name, vae, prompt, neg_prompt, random.choice(sampler),
                    s, cfg__, den__, batch_size, width=816, height=1088
                )
                for im in imgs:
                    im = im.resize((408, 544))
                    im.save(
                        os.path.join(
                            out_path, model_name[:5] + "_" + str(s) + "_" + str(cfg__) + "_" + str(den__) + "_"
                                      + datetime.now().strftime("%d %H %M") + f"_{str(uuid4())[:4]}.png"
                        )
                    )
    except:
        time.sleep(10)
