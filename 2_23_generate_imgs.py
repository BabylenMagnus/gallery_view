import os
import random
import json

import pandas as pd
import numpy as np

from uuid import uuid4
from tqdm import tqdm

import torch

from diffusers import SemanticStableDiffusionPipeline, DDIMScheduler


IMGS_PATH = "OUTPUT/2_21_output"
OUTPUT_PATH = r"OUTPUT/2_23_OUTPAINT"
INPUT_CSV = r"data/2_21_without_image.csv"

OUT_PATH = r"Z:\Coding\gallery_view\OUTPUT"

sd_models = [
    "SdValar/deliberate2",
    "digiplay/DreamShaper_8",
    "digiplay/RealCartoon3D_F16full_v3.1",
    "xiaolxl/GuoFeng3",
    "danbrown/RevAnimated-v1-2-2",
    "emilianJR/epiCRealism",
    "sinkinai/Babes-2.0",
    "Yntec/a-ZovyaRPGArtistV2VAE",
    "Lykon/DreamShaper",
    "ehristoforu/Fluently",
    "xiaolxl/GuoFeng3",
    "shibing624/asian-role",
    "emilianJR/chilloutmix_NiPrunedFp32Fix"
]


NEG_PROMPT = ("(deformed iris, deformed pupils), text, worst quality, low quality, jpeg artifacts, ugly, "
              "morbid, mutilated, (extra fingers), (mutated hands), poorly drawn hands, poorly drawn face, mutation, "
              "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
              "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
              "(fused fingers), (too many fingers), long neck, (bad-prompt,bad-quality,bad-artist,bad-hands-5,"
              "bad-hand-V4,bad-image-v2-39000,highway,road, bad-picture-chill-75v:1.4),Easy-Negative,ely-neg-prompt3,"
              "ng-deepnegative-V1-75t,borrowed character, schizo-neg-prompt,test-neg2,verybadimagenegative-V1.2-6400,"
              "negative-hand-neg,umbrella,building, (((Text Signature, signatures, authorship, author:1.3))), "
              "(deformed iris, deformed pupils), text, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
              "morbid, mutilated, (extra fingers), (mutated hands), poorly drawn hands, poorly drawn face, mutation, "
              "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
              "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
              "(fused fingers), (too many fingers), long neck, (deformed iris, deformed pupils), text, worst quality, "
              "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (extra fingers), (mutated hands), "
              "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
              "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, "
              "missing arms, missing legs, extra arms, extra legs, (fused fingers), (too many fingers), long neck")


def get_prompt_embeddings(
        pipe,
        prompt,
        negative_prompt,
        split_character=",",
        device=torch.device("cuda")
):
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
            return_tensors="pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False
        ).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            max_length=shape_max_length
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


pipe = None


def load_model(model_name: str):
    global pipe
    del pipe

    pipe = SemanticStableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        requires_safety_checker=False,
        safety_checker=None,
        device_map=None,
        low_cpu_mem_usage=False
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe


def infer(pipe, prompt):
    # prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
    #     pipe,
    #     prompt,
    #     NEG_PROMPT,
    #     split_character=",",
    #     device="cuda"
    # )
    images = pipe(
        prompt=prompt, negative_prompt=NEG_PROMPT,
        guidance_scale=7, num_inference_steps=40, num_images_per_prompt=BATCH_SIZE, height=560, width=424
    ).images

    return images


out_dict = {}
with open(os.path.join(OUT_PATH, "prompts.json"), "r") as t:
    out_dict = json.load(t)
    print(len(out_dict))

data = pd.read_csv(INPUT_CSV)

N_ITER = 2
BATCH_SIZE = 15

local_samplers = ["DDIM", "DPM++ 2M Karras", "Euler a", "DPM++ SDE Karras"]

model_name = random.choice(sd_models)
pipe = load_model(model_name)

LIST = os.listdir(IMGS_PATH)
np.random.shuffle(LIST)

N = 0
J = 0
K = 0

for img_path in tqdm(LIST):
    prompt_name = data[data["Hash"] == img_path[:-4]]["Name"].values[0]
    prompt_blip = out_dict[img_path]

    out_path = os.path.join(OUTPUT_PATH, img_path[:-4])
    if os.path.exists(out_path) and len(os.listdir(out_path)) > 35:
        J += 1
        print("Shit ", J)
        continue
    if os.path.exists(out_path) and random.random() < 0.8:
        K += 1
        print("Common ", K)
        continue
    elif os.path.exists(out_path):
        N += 1
        print("Еее ", N)

    os.makedirs(out_path, exist_ok=True)

    if random.random() > 0.8:
        model_name = random.choice(sd_models)
        pipe = load_model(model_name)

    for prompt in [
        "russian girl, cute girl, " + prompt_name,
        "awesome girl, cute girl, " + prompt_blip,
        prompt_name + ", " + prompt_blip
    ]:
        res_imgs = infer(pipe, prompt)

        for img_ in res_imgs:
            img_.save(os.path.join(
                out_path,
                f"{model_name.split('/')[1]}_{str(uuid4())[:3]}.png"
            ))
