from transformers import Blip2Processor, Blip2ForConditionalGeneration

from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np

import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    load_in_8bit=True, device_map="auto",
    resume_download=True
)
blip.eval()


def load_img_to_array(img_p):
    img = Image.open(img_p)
    img = img.convert("RGB")
    return np.array(img)


PATH = "/home/jjjj/Documents/gallery_view/IMGS_22.11/without"
OUT_PATH = "/home/jjjj/Documents/gallery_view/IMGS_22.11"

with open(os.path.join(OUT_PATH, "prompts.json"), "r") as t:
    out_dict = json.load(t)

# out_dict = {}

for img_path in tqdm(os.listdir(PATH)):
    if not (img_path.endswith(".jpg") or img_path.endswith(".png")):
        continue
    if img_path[:-4] in out_dict:
        continue

    img = load_img_to_array(os.path.join(PATH, img_path))
    inputs = processor(Image.fromarray(img), None, return_tensors="pt").to("cuda", torch.float16)
    out = blip.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True).strip()
    out_dict[img_path[:-4]] = str(prompt)

with open(os.path.join(OUT_PATH, "prompts.json"), "w") as t:
    json.dump(out_dict, t, indent=2)
