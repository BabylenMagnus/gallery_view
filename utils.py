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
A111_url = "http://127.0.0.1:7860"
BASE_NEGATIV_PROMPT = \
    ("poster, text, logo of video game, poster art, concert poster, the logo for the video game, the word sipop on it, "
     "logo, poster art")
SIZES = {
    "Главный 1174 х 360": [360, 1174],
    "Главный 698 х 360": [360, 698],
    "Внутряк 1014 х 360": [360, 1014],
    "Форма логина 640 х 80": [80, 640],
    "Мобила cтраницы 1242 х 480": [480, 1242],
    "Мобила форма логина 1194 х 288": [288, 1194],
    "Стандарт 408×544": [544, 408],
}


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
        draw.rectangle(((x1, y1), (x2, y2)))

    return new_img, img, bboxes, np.array(res)


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


def remove_text(
        img, bboxes, map_bboxes, prompt, negative_prompt, model, sampler, steps, cfg_scale, denoising_strength
):
    b = []
    for bb, i in zip(bboxes, map_bboxes):
        if i:
            b.append(bb)

    frame_around_size = 10
    mask = Image.new("L", (img.shape[1], img.shape[0]))
    draw = ImageDraw.Draw(mask)

    for i in b:
        i = np.array(i)
        i[0, :] -= frame_around_size
        i[1, :] += frame_around_size
        draw.rectangle([tuple(x) for x in i.tolist()], fill="white")

    mask = np.array(mask).astype(np.uint8)

    Image.fromarray(mask).save("mask_delete_text.png")
    Image.fromarray(img).save("img_delete_text.png")

    return generate_image(img, mask, model, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength)


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
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    jpg_img = cv2.imencode('.png', img)
    b64_string = base64.b64encode(jpg_img[1]).decode("utf-8")
    return b64_string


def generate_image(img, mask, model, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": 1,
        "include_init_images": True,
        "mask": to_b64(mask),
        "init_images": [
            to_b64(img)
        ],
        "inpainting_fill": 0,
        "override_settings": {
            "sd_model_checkpoint": model
        },
        "width": img.shape[1],
        "height": img.shape[0],
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength
    }

    response = requests.post(f'{A111_url}/sdapi/v1/img2img', json=payload)
    return Image.open(io.BytesIO(base64.b64decode(response.json()["images"][0])))


def inpaint_image(img, model, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength):
    mask = img["mask"]
    img = img["image"]
    Image.fromarray(mask).save("mask_inp.png")
    Image.fromarray(img).save("img_inp.png")

    return generate_image(img, mask, model, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength)


def get_models():
    return [i['title'] for i in requests.get(f'{A111_url}/sdapi/v1/sd-models').json()]


def get_samplers():
    return [i["name"] for i in requests.get(url=f'http://127.0.0.1:7860/sdapi/v1/samplers').json()]


def outpainting(
        img, model, left, top, right, bottom, h, w, prompt,
        negative_prompt, sampler, steps, cfg_scale, denoising_strength
):
    h = int(h)
    w = int(w)
    if img.shape[0] > h or img.shape[1] > w:
        percentage = min(w / img.shape[1], h / img.shape[0])
        new_w = int(percentage * img.shape[1])
        new_h = int(percentage * img.shape[0])
        img = np.array(Image.fromarray(img).resize((new_w, new_h), reducing_gap=3))

    n = 5

    new_img = np.zeros([h, w, 3], dtype=np.uint8)

    if top != bottom:
        top = 0 if top else new_img.shape[0] - img.shape[0]
        bottom = new_img.shape[0] if bottom else img.shape[0]
    if right != left:
        left = 0 if left else new_img.shape[1] - img.shape[1]
        right = new_img.shape[1] if right else img.shape[1]

    if right == left:
        left = (new_img.shape[1] - img.shape[1]) // 2
        right = left + img.shape[1]
    if top == bottom:
        top = (new_img.shape[0] - img.shape[0]) // 2
        bottom = top + img.shape[0]

    new_img[top:bottom, left:right, :] = img

    mask = np.ones([new_img.shape[0], new_img.shape[1], 3], dtype=np.uint8) * 255
    mask[top + n:bottom - n, left + n:right - n, :] = 0
    mask = mask.astype(np.uint8)
    return generate_image(new_img, mask, model, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength)


def outpainting_with_value(
        img, model, left, top, right, bottom, size, prompt,
        negative_prompt, sampler, steps, cfg_scale, denoising_strength
):
    h, w = SIZES[size]
    return outpainting(
        img, model, left, top, right, bottom, h, w, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength
    )


def add_text_to_xy(img, text, x, y, font_choose, font_size, color):
    pil_img = Image.fromarray(img)
    font = ImageFont.truetype(os.path.join(FONT_PATH, font_choose), size=int(font_size))

    image = Image.new('RGB', pil_img.size, "white")
    draw = ImageDraw.Draw(image)

    _, _, w, h = draw.textbbox(
        (0, 0), text, font=font
    )

    # Extract the selected coordinates from the event data
    coord = (x - w // 2, y - h // 2)
    draw = ImageDraw.Draw(pil_img)

    # Add text to the selected coordinate
    draw.text(coord, text, font=font, fill=color)
    img_with_text = np.array(pil_img)

    return img_with_text


def add_text_to_coords(img, text, evt: gr.SelectData, font_choose, font_size, color):
    return (
        add_text_to_xy(img, text, evt.index[0], evt.index[1], font_choose, font_size, color), evt.index[0], evt.index[1]
    )


def save_text(input_img, text_input, x_cord, y_cord, font_choose, font_size, color, history):
    if history is None:
        history = []
    history.append([text_input, x_cord, y_cord, font_choose, font_size, color])
    return add_text_to_xy(input_img, *history[-1]), history


def undo(history, img):
    if history is None or not len(history):
        return None, img

    history.pop()

    for args in history:
        img = add_text_to_xy(img, *args)
    return history, img


FONT_PATH = "fonts/"
MODELS = get_models()
SAMPLERS = get_samplers()
