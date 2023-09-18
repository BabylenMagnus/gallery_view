from ctrnet_infer import CTRNetInfer
import gradio as gr

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import easyocr

import os
import io
import requests
import base64
import cv2


eader = easyocr.Reader(['en'])
FONT_PATH = "fonts/"
A111_url = "http://127.0.0.1:7860"
BASE_NEGATIV_PROMPT = \
    ("poster, text, logo of video game, poster art, concert poster, the logo for the video game, the word sipop on it, "
     "logo, poster art")

# model_path = "models/CTRNet_G.onnx"
# ctrnet = CTRNetInfer(model_path)


def ocr_detect(img):
    new_img = Image.fromarray(img.copy())
    result = eader.readtext(np.array(new_img))
    new_ = []

    res = []
    for i in result:
        i = np.array(i[0])
        new_.append([(i[:, 0].min(), i[:, 1].min()), (i[:, 0].max(), i[:, 1].max())])
        res.append([
            (i[:, 0].min(), i[:, 1].min()),
            (i[:, 0].min(), i[:, 1].max()),
            (i[:, 0].max(), i[:, 1].max()),
            (i[:, 0].max(), i[:, 1].min())
        ])

    if len(new_) == 0:
        return None, None

    bboxes = np.array(new_).tolist()

    draw = ImageDraw.Draw(new_img)

    for (x1, y1), (x2, y2) in bboxes:
        draw.rectangle([(x1, y1), (x2, y2)])

    return new_img, img, bboxes, np.array(res), result


def choose_bboxes(img, bboxes, map_bboxes, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    if map_bboxes is None:
        map_bboxes = [False for _ in bboxes]

    img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(img)

    for i, ((x1, y1), (x2, y2)) in enumerate(bboxes):
        if x1 < x < x2 and y1 < y < y2:
            map_bboxes[i] = True
        if map_bboxes[i]:
            draw.rectangle([(x1, y1), (x2, y2)], outline="red")

    return img, bboxes, map_bboxes


def remove_text(img, bboxes, map_bboxes, result):
    b = []
    for bb, i in zip(bboxes, map_bboxes):
        if i:
            b.append(bb)

    n = max(img.shape[:2])

    new_img = np.zeros((n, n, 3), dtype=np.uint8)
    new_img[:img.shape[0], :img.shape[1], :] += img

    pred = ctrnet(new_img, np.array(b))

    data = {}
    data["text"] = []
    data["top"] = []
    data["left"] = []
    data["height"] = []

    for r, i in zip(result, map_bboxes):
        if i:
            data["text"].append(r[1])
            r = np.array(r[0])
            data["top"].append(r[:, 1].min())
            data["left"].append(r[:, 0].min())
            data["height"].append(r[:, 1].max() - r[:, 1].min())

    return pred[:img.shape[0], :img.shape[1], :], pd.DataFrame(data)


def add_text_font(img, result, font_name, color):
    img = Image.fromarray(img)
    for j in range(len(result)):
        j = result.loc[j]
        font = ImageFont.truetype(
            os.path.join(FONT_PATH, font_name),
            int(j["height"])
        )
        draw = ImageDraw.Draw(img)
        draw.text((int(j["left"]), int(j["top"])), j["text"], font=font, fill=color)
    return img


def to_b64(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    jpg_img = cv2.imencode('.png', img)
    b64_string = base64.b64encode(jpg_img[1]).decode("utf-8")
    return b64_string


def inpaint_image(img, model, prompt, negative_prompt):
    mask = img["mask"]
    img = img["image"]

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 40,
        "batch_size": 1,
        "include_init_images": True,
        "mask": to_b64(mask),
        "init_images": [
            to_b64(img)
        ],
        "inpainting_fill": 0,
        "override_settings": {
            "sd_model_checkpoint": model
        }
    }

    response = requests.post(f'{A111_url}/sdapi/v1/img2img', json=payload)
    return Image.open(io.BytesIO(base64.b64decode(response.json()["images"][0])))


def get_models():
    return [i['title'] for i in requests.get(f'{A111_url}/sdapi/v1/sd-models').json()]


