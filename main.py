import json
import os
import pandas as pd
import numpy as np
import cv2

import gradio as gr
from PIL import Image, ImageFont, ImageDraw
import glob


RESULT_PATH = "/home/jjjj/Documents/Inpaint-Anything/results_8/"
BBOX_PATH = "/home/jjjj/Pictures/without_/bboxes.json"

ORIG_IMG = "/home/jjjj/Documents/new"

log_path = "log.txt"


with open(BBOX_PATH, "r") as t:
    BBOXES = json.load(t)

RESULT_DIRS = [i for i in os.listdir(RESULT_PATH) if os.path.isdir(os.path.join(RESULT_PATH, i))]


def get_imgs(num):
    return [Image.open(i) for i in glob.glob(RESULT_PATH + RESULT_DIRS[int(num)] + "/*.jpg")]


def change_dir(num, button=None):
    if button is None:
        return get_imgs(num), "# " + RESULT_DIRS[int(num)]
    if button == "+" and num < len(RESULT_DIRS):
        num += 1
    elif num > 0:
        num -= 1
    return get_imgs(num), num, "# " + RESULT_DIRS[int(num)]


def select_image(evt: gr.SelectData, gallery):
    return Image.open(gallery[evt.index]['name'])


def load_img_to_array(img_p):
    img = Image.open(img_p)
    img = img.convert("RGB")
    return np.array(img)


def add_text(img, page):
    orig_img = load_img_to_array(os.path.join(ORIG_IMG, RESULT_DIRS[int(page)] + ".jpg"))

    a = Image.fromarray(orig_img).convert("L")
    a = np.array(a)
    img = np.array(Image.fromarray(img.copy()).resize((a.shape[1], a.shape[0])))

    blured = cv2.GaussianBlur(a, (1, 1), cv2.BORDER_DEFAULT)

    mask = np.zeros([orig_img.shape[0], orig_img.shape[1]], dtype=np.uint8)
    mask[310:][blured[310:] > 195] = 255

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(mask, [c], -1, 255, thickness=3)

    mask = np.stack([mask, mask, mask], axis=2)

    img[mask > 1] = orig_img[mask > 1]

    with open(log_path, "r") as t:
        data = t.read().split("\n")

    data.append(RESULT_DIRS[int(page)])
    data = list(set(data))

    with open(log_path, "w") as t:
        t.write("\n".join(data))

    return img


# def add_text(img, result):
#     print(len(result))
#
#     img = Image.fromarray(img)
#     for j in range(len(result)):
#         j = result.loc[j]
#         h = j["bottom"] - j["top"]
#         w = j["right"] - j["left"]
#
#         image = Image.new('L', (int(w * 2), int(h * 2)), "white")
#
#         draw = ImageDraw.Draw(image)
#         myFont = ImageFont.truetype("/home/jjjj/Downloads/Blacknorthdemo-mLE25.otf", size=h)
#         _, _, w_, h_ = draw.textbbox((0, 0), j["name"], font=myFont)
#         draw.text((0, 0), j["name"], font=myFont, fill="black")
#         text_ = np.array(image)[:h_, :w_]
#         text_ = Image.fromarray(text_).resize((int(w), int(h)), reducing_gap=3.0)
#         img.paste(text_, (j["left"], j["bottom"]), mask=Image.fromarray(255 - np.array(text_)))
#
#     return img


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Choose Image"):
            with gr.Row():
                prev_page = gr.Button(
                    "prev", variant="primary"
                )
                page = gr.Number(value=0, minimum=0, maximum=len(RESULT_DIRS), show_label=False)
                next_page = gr.Button(
                    "next", variant="primary"
                )
                dir_name = gr.Markdown(value="# " + RESULT_DIRS[int(page.value)])

            gallery = gr.Gallery(
                value=get_imgs(page.value), object_fit="contain",
                label="Generated images", show_label=False, elem_id="gallery", columns=9, height=850
            )

        with gr.TabItem("Add text"):
            with gr.Row():
                image = gr.Image(height=800, show_download_button=True)
                with gr.Column():
                    # result = gr.Dataframe(
                    #     headers=["name", "left", "right", "top", "bottom"],
                    #     datatype=["str", "number", "number", "number", "number"]
                    # )
                    add_text_to_image = gr.Button("add text", variant="primary")

    page.submit(
        change_dir,
        [page],
        [gallery, dir_name]
    )

    prev_page.click(
        lambda page: change_dir(page, "-"),
        [page],
        [gallery, page, dir_name]
    )

    next_page.click(
        lambda page: change_dir(page, "+"),
        [page],
        [gallery, page, dir_name]
    )

    gallery.select(
        select_image,
        [gallery],
        [image]
    )

    add_text_to_image.click(
        add_text,
        [image, page],
        [image]
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5000)
