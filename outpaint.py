from utils import outpainting, INP_MODELS, VAES, BASE_NEGATIV_PROMPT

import os
import random
import json

from PIL import Image
import numpy as np
import pandas as pd

from uuid import uuid4
from tqdm import tqdm


IMGS_PATH = "OUTPUT/2_21_output"
OUTPUT_PATH = r"C:\Users\user\Documents\2_23_OUTPAINT"
INPUT_CSV = r"data/2_21_without_image.csv"


OUT_PATH = r"Z:\Coding\gallery_view\OUTPUT"

out_dict = {}
with open(os.path.join(OUT_PATH, "prompts.json"), "r") as t:
    out_dict = json.load(t)
    print(len(out_dict))

data = pd.read_csv(INPUT_CSV)

N_ITER = 2
BATCH_SIZE = 10

local_samplers = ["DDIM", "DPM++ 2M Karras", "Euler a", "DPM++ SDE Karras"]

model = random.choice(INP_MODELS)
vae = random.choice(VAES)

LIST = os.listdir(IMGS_PATH)
np.random.shuffle(LIST)

N = 0

for img_path in tqdm(LIST):
    prompt_name = data[data["ID"] == img_path[:-4]]["Name"].values[0]
    prompt_blip = out_dict[img_path]

    img_path = random.choice(os.listdir(IMGS_PATH))

    out_path = os.path.join(OUTPUT_PATH, img_path[:-4])
    n = random.random()
    if os.path.exists(out_path) and n < 0.85:
        continue
    elif os.path.exists(out_path):
        N += 1
        print("Еее ", N)

    os.makedirs(out_path, exist_ok=True)

    img = np.array(Image.open(os.path.join(IMGS_PATH, img_path)).convert("RGB"))
    try:
        img = img[15:-15, 15:-15, :]
    except:
        print("error on ", img_path)
        continue
    h, w = img.shape[:-1]
    h = int(w * 1.34)

    if random.random() > 0.8:
        model = random.choice(INP_MODELS)
        vae = random.choice(VAES)

    for _ in range(N_ITER):
        sampler = random.choice(local_samplers)
        steps = random.randint(30, 50)
        cfg_scale = random.choice([3.5, 4, 5, 6, 7])
        denoising_strength = random.choice([0.5, 0.6, 0.75])

        prompt = random.choice([prompt_name, prompt_blip, prompt_name + ", " + prompt_blip])

        res_imgs = outpainting(
            img, model, vae, False, False, False, False, h, w, prompt, BASE_NEGATIV_PROMPT,
            sampler, steps, cfg_scale, denoising_strength=denoising_strength, batch_size=BATCH_SIZE
        )

        for img_ in res_imgs:
            img_.save(
                os.path.join(
                    out_path,
                    f"{model[:10]}_{sampler[:5]}_{steps}_{cfg_scale}_{denoising_strength}_{str(uuid4())[:3]}.png"
                )
            )
