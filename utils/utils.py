import gradio as gr

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import easyocr

import os
import io
import requests
import base64
import cv2

from datetime import datetime
from uuid import uuid4


eader = easyocr.Reader(['en'])
A111_url = "http://127.0.0.1:7860"
BASE_NEGATIV_PROMPT = (
    "worst quality, low quality, normal quality, lowres, ugly, bad, sketch, text, jpeg, artifacts, room, wall,"
    " wallpaper, floor, gradient, poster, banner, blurring, fog, smoke, blur, monochrome background, monochrome, "
    "background"
)
# BASE_NEGATIV_PROMPT = (
#     "two straws, sketches, worst quality, low quality, normal quality, lowres, normal quality, freckles, ugly, bad, "
#     "sketch, bad anatomy, out of view, cut off, ugly, deformed, mutated, skin spots, skin spots, acnes, skin blemishes,"
#     "extra fingers,fewer fingers, (ugly eyes, deformed iris, deformed pupils, fused lips and teeth:1.2), text, jpeg "
#     "artifacts, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry,"
#     " dehydrated, bad anatomy, bad proportions"
# )

SIZES = {
    "Главный 1174 х 360": [360, 1174],
    "Главный 698 х 360": [360, 698],
    "Внутряк 1014 х 360": [360, 1014],
    "Форма логина 640 х 80": [80, 640],
    "Мобила cтраницы 1242 х 480": [480, 1242],
    "Мобила форма логина 1194 х 288": [288, 1194],
    "Стандарт 408×544": [544, 408],
}


def log_img(imgs):
    for img in imgs:
        img.save(os.path.join("../log", f"{str(uuid4())[:8]}.png"))


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
        return img, img, None, None, None

    bboxes = np.array(new_).tolist()

    draw = ImageDraw.Draw(new_img)

    for (x1, y1), (x2, y2) in bboxes:
        draw.rectangle(((x1, y1), (x2, y2)))

    return new_img, img, bboxes, np.array(res), None


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
        img, bboxes, map_bboxes, prompt, negative_prompt, model, vae_name,
        sampler, steps, cfg_scale, denoising_strength, frame_around_size=10, batch_size=1
):
    b = []
    for bb, i in zip(bboxes, map_bboxes):
        if i:
            b.append(bb)
    print(b)

    mask = Image.new("L", (img.shape[1], img.shape[0]))
    draw = ImageDraw.Draw(mask)
    frame_around_size = int(frame_around_size)

    for i in b:
        i = np.array(i)
        i[0, :] -= frame_around_size
        i[1, :] += frame_around_size
        draw.rectangle([tuple(x) for x in i.tolist()], fill="white")

    mask = np.array(mask).astype(np.uint8)

    Image.fromarray(mask).save("mask_delete_text.png")
    Image.fromarray(img).save("img_delete_text.png")

    return generate_image(
        img, mask, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
    )


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


def generate_image_txt2img(
        model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size=1
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": batch_size,
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae_name
        },
        "width": 408,
        "height": 544,
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength
    }

    response = requests.post(f'{A111_url}/sdapi/v1/txt2img', json=payload)
    imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in response.json()["images"]]
    log_img(imgs)
    return imgs


def generate_image(
        img, mask, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size=1
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": batch_size,
        "include_init_images": True,
        "mask": to_b64(mask),
        "init_images": [
            to_b64(img)
        ],
        "inpainting_fill": 0,
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae_name
        },
        "width": img.shape[1],
        "height": img.shape[0],
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength
    }

    response = requests.post(f'{A111_url}/sdapi/v1/img2img', json=payload)
    imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in response.json()["images"]]
    log_img(imgs)
    return imgs


def generate_image_upgrade(
        img, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size=1,
        width=None, height=None
):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": batch_size,
        "init_images": [
            to_b64(img)
        ],
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae_name
        },
        "width": img.shape[1] if width is None else width,
        "height": img.shape[0] if height is None else height,
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength
    }

    response = requests.post(f'{A111_url}/sdapi/v1/img2img', json=payload)
    imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in response.json()["images"]]
    log_img(imgs)
    return imgs


def inpaint_image(img, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size):
    mask = img["mask"]
    img = img["image"]
    Image.fromarray(mask).save("mask_inp.png")
    Image.fromarray(img).save("img_inp.png")

    return generate_image(
        img, mask, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
    )


def get_models():
    return [i['title'] for i in requests.get(f'{A111_url}/sdapi/v1/sd-models').json()]


def get_vaes():
    return [i['model_name'] for i in requests.get(f'{A111_url}/sdapi/v1/sd-vae').json()]


def get_samplers():
    return [i["name"] for i in requests.get(url=f'{A111_url}/sdapi/v1/samplers').json()]


def outpainting(
        img, model, vae_name, left, top, right, bottom, h, w, prompt,
        negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
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
    return generate_image(
        new_img, mask, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
    )


def outpainting_with_value(
        img, model, vae_name, left, top, right, bottom, size, prompt,
        negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
):
    h, w = SIZES[size]
    return outpainting(
        img, model, vae_name, left, top, right, bottom, h, w, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size
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


def get_model_controlnet(name):
    a = requests.get(url=f'{A111_url}/controlnet/model_list').json()
    for i in a["model_list"]:
        if name in i:
            return i


def controlnet_preview(module_name, img, x=64, y=64):
    payload = {
        "controlnet_module": module_name,
        "controlnet_input_images": [to_b64(img)],
        "controlnet_processor_res": max(img.shape),
        "controlnet_threshold_a": x,
        "controlnet_threshold_b": y
    }
    a = requests.post(url=f'{A111_url}/controlnet/detect', json=payload).json()
    return Image.open(io.BytesIO(base64.b64decode(a["images"][0])))


def controlnet_generate_txt(
        module_controlnet, model_controlnet, img, model, vae_name, prompt, negative_prompt, sampler, steps,
        cfg_scale, denoising_strength, batch_size, guidance_start, guidance_end, control_mode, lora_add_detail,
        lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x=64, y=64
):
    if lora_add_detail:
        prompt += f" <lora:add_detail:{lora_add_detail_value}>"
    if lora_add_details:
        prompt += f" <lora:more_details:{lora_add_details_value}>"
    if lora_blindbox:
        prompt += f" <lora:3DMM_V12:1>"
    if lora_eyes_gen:
        prompt += f" <lora:more_details:{lora_eyes_gen_value}>"
    if lora_polyhedron_fem:
        prompt += f" <lora:polyhedron_all_eyes:{lora_polyhedron_fem_value}>"
    if lora_polyhedron_man:
        prompt += f" <lora:polyhedron_men_eyes:{lora_polyhedron_man_value}>"
    if lora_beautiful_detailed:
        prompt += f" <lora:BeautifulDetailedEyes:{lora_beautiful_detailed_value}>"

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": batch_size,
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae_name
        },
        "width": img.shape[1],
        "height": img.shape[0],
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": to_b64(img),
                        "module": module_controlnet,
                        "processor_res": max(img.shape),
                        "threshold_a": x,
                        "threshold_b": y,
                        "guidance_start": guidance_start,
                        "guidance_end": guidance_end,
                        "control_mode": CONTROL_MODE.index(control_mode),
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }
    if model_controlnet:
        payload["alwayson_scripts"]["controlnet"]["args"][0]["model"] = model_controlnet

    response = requests.post(f'{A111_url}/sdapi/v1/txt2img', json=payload)
    imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in response.json()["images"]]
    log_img(imgs)
    if len(imgs) == batch_size:
        return imgs
    return imgs[:-1], imgs[-1]


def controlnet_generate(
        module_controlnet, model_controlnet, img, model, vae_name, prompt, negative_prompt, sampler, steps,
        cfg_scale, denoising_strength, batch_size, guidance_start, guidance_end, control_mode, lora_add_detail,
        lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x=64, y=64,
        width=None, height=None
):
    if lora_add_detail:
        prompt += f" <lora:add_detail:{lora_add_detail_value}>"
    if lora_add_details:
        prompt += f" <lora:more_details:{lora_add_details_value}>"
    if lora_blindbox:
        prompt += f" <lora:3DMM_V12:1>"
    if lora_eyes_gen:
        prompt += f" <lora:more_details:{lora_eyes_gen_value}>"
    if lora_polyhedron_fem:
        prompt += f" <lora:polyhedron_all_eyes:{lora_polyhedron_fem_value}>"
    if lora_polyhedron_man:
        prompt += f" <lora:polyhedron_men_eyes:{lora_polyhedron_man_value}>"
    if lora_beautiful_detailed:
        prompt += f" <lora:BeautifulDetailedEyes:{lora_beautiful_detailed_value}>"

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "batch_size": batch_size,
        "include_init_images": True,
        "init_images": [
            to_b64(img)
        ],
        "override_settings": {
            "sd_model_checkpoint": model,
            "sd_vae": vae_name
        },
        "width": img.shape[1] if width is None else width,
        "height": img.shape[0] if height is None else height,
        "sampler_index": sampler,
        "cfg_scale": cfg_scale,
        "denoising_strength": denoising_strength,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": to_b64(img),
                        "module": module_controlnet,
                        "processor_res": max(img.shape),
                        "threshold_a": x,
                        "threshold_b": y,
                        "guidance_start": guidance_start,
                        "guidance_end": guidance_end,
                        "control_mode": CONTROL_MODE.index(control_mode),
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }
    if model_controlnet:
        payload["alwayson_scripts"]["controlnet"]["args"][0]["model"] = model_controlnet

    response = requests.post(f'{A111_url}/sdapi/v1/img2img', json=payload)
    imgs = [Image.open(io.BytesIO(base64.b64decode(i))) for i in response.json()["images"]]
    log_img(imgs)
    if len(imgs) == batch_size:
        return imgs
    return imgs[:-1], imgs[-1]


def canny_preview(img, x, y):
    return controlnet_preview("canny", img, x, y)


def depth_preview(img, type, x, y):
    return controlnet_preview(type, img, x, y)


def normal_preview(img, type, x):
    return controlnet_preview(type, img, x)


def pose_preview(img, type):
    return controlnet_preview(type, img)


def canny_generate(
        img, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size_cn,
        guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value, lora_add_details,
        lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem,
        lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value, lora_beautiful_detailed,
        lora_beautiful_detailed_value, x, y
):

    return controlnet_generate(
        "canny", get_model_controlnet("canny"), img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x, y
    )


def depth_generate(
        img, depth_type, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value, x, y
):

    return controlnet_generate(
        depth_type, get_model_controlnet("depth"), img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x, y
    )


def normal_generate(
        img, depth_type, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value, x
):

    return controlnet_generate(
        depth_type, get_model_controlnet("normal"), img, model, vae_name, prompt, negative_prompt, sampler, steps,
        cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail,
        lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x
    )


def pose_generate(
        img, depth_type, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value, x
):

    return controlnet_generate(
        depth_type, get_model_controlnet("pose"), img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, x
    )


def line_generate(
        img, depth_type, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value
):
    cn_name = get_model_controlnet("lineart_anime") if "anime" in depth_type else get_model_controlnet("lineart")

    return controlnet_generate(
        depth_type, cn_name, img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
    )


def shuffle_generate(
        img, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value
):
    return controlnet_generate(
        "shuffle", get_model_controlnet("shuffle"), img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
    )


def reference_generate(
        img, reference_type, model, vae_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength,
        batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail, lora_add_detail_value,
        lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen, lora_eyes_gen_value,
        lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man, lora_polyhedron_man_value,
        lora_beautiful_detailed, lora_beautiful_detailed_value
):
    return controlnet_generate(
        reference_type, "", img, model, vae_name, prompt, negative_prompt,
        sampler, steps, cfg_scale, denoising_strength, batch_size_cn, guidance_start, guidance_end, control_mode,
        lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
        lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
        lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
    )


FONT_PATH = "../fonts/"
MODELS = [i for i in get_models() if "inp" not in i.lower()]
INP_MODELS = [i for i in get_models() if "inp" in i.lower()]
CONTROL_MODE = ["Balanced", "My prompt is more important", "ControlNet is more important"]
VAES = ["None"] + get_vaes()
SAMPLERS = get_samplers()
