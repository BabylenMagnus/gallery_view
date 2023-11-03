import os
import pandas as pd
import numpy as np
import cv2

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import glob
import json

import random


RESULT_PATH = "log"
ORIG_IMG = "/root/gallery_view/results/"

INPUT_EXCEL = f"/home/jjjj/Downloads/Картинки .xlsx"

xl = pd.ExcelFile(INPUT_EXCEL)
data = xl.parse(xl.sheet_names[0])
SLOTS_NAMES = list(set(list(data['name'])))


save_path = "saves/"
log_path = "log.txt"


def get_imgs():
    pathes = random.choices(glob.glob(RESULT_PATH + "/*.png"), k=30)

    return [Image.open(i) for i in pathes], pathes


def change_dir(num, button=None):
    print(num)
    if button is None:
        return num, SLOTS_NAMES[int(num)]
    if button == "+" and num < len(SLOTS_NAMES):
        num += 1
    elif num > 0:
        num -= 1
    return num, SLOTS_NAMES[int(num)]


def split_text(text):
    THRESH = 8
    THRESH_2 = 7

    new_text = []

    if text[0] == "THE":
        del text[0]
        text[0] = "THE " + text[0]

    if "VS" in text:
        text = " ".join(text)
        text = text.split(" VS ")
        return [[text[0]], ["VS"], [text[1]]]

    if len(text) <= 3:
        if len(" ".join(text)) < THRESH_2:
            new_text = [[" ".join(text)]]
        else:
            new_text = [text[:1], text[1:]]
    else:
        a = ""
        for i in text:
            if len(a) < THRESH:
                a += " " + i
            else:
                new_text.append([a.strip()])
                a = i
        new_text.append([a.strip()])
    return new_text


splits = {
    1: [[340, 78]],
    2: [[355, 56], [400, 68]],
    3: [[310, 68], [370, 44], [420, 68]]
}

UP_FONT_PATH = "/home/jjjj/Documents/gallery_view/fonts/RubikWetPaint-Regular.ttf"

FONT_PATH = "/home/jjjj/Documents/gallery_view/fonts/RedHatText-VariableFont_wght.ttf"

TOP = 490
tops = [310, 350, 350, 340]


def select_image(evt: gr.SelectData, gallery, page):
    text = split_text(SLOTS_NAMES[int(page)].split(" "))

    img = Image.open(gallery[evt.index]['name'])
    img = img.convert("RGB")

    s = splits[len(text)]

    top = tops[len(text)]

    for t in text:
        t = " ".join(t)
        image = Image.new('RGB', img.size, "white")
        draw = ImageDraw.Draw(image)

        w = 1000
        for f_size in [38, 42, 46, 48, 52, 56, 64, 72, 80, 98, 106, 112]:
            font = ImageFont.truetype(UP_FONT_PATH, f_size)

            _, _, w, h = draw.textbbox(
                (0, 0), t, font=font
            )
            if w > 360 or h > (150 / len(text)):
                draw = ImageDraw.Draw(img)
                draw.text(
                    ((img.size[0] - w) / 2, top), t, font=font, fill="white", stroke_width=1, stroke_fill="black"
                )
                top += h - 10
                break

    return img, evt.index


def load_img_to_array(img_p):
    img = Image.open(img_p)
    img = img.convert("RGB")
    return np.array(img)


def save_img(img, page):
    img = Image.fromarray(img)
    name = RESULT_DIRS[int(page)]
    print("save", name)
    img.save(save_path + name + "__" + str(len(glob.glob(save_path + name + "*"))) + ".png")


with gr.Blocks() as demo:
    pathes = gr.State(None)
    text = gr.State(None)
    index_path = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Choose Image"):
            with gr.Row():
                gallery = gr.Gallery(
                    object_fit="contain", label="Generated images",
                    show_label=False, elem_id="gallery", columns=5, height=850
                )

                with gr.Column():
                    start = gr.Button("Start")
                    with gr.Row():
                        prev_page = gr.Button(
                            "prev", variant="primary"
                        )
                        page = gr.Number(value=0, minimum=0, maximum=len(SLOTS_NAMES), show_label=False)
                        next_page = gr.Button(
                            "next", variant="primary"
                        )

                    image = gr.Image(height=544, width=408, show_download_button=False)
                    save_button = gr.Button("Save image", variant="primary")

            # with gr.Row():
            #     prev_page = gr.Button(
            #         "prev", variant="primary"
            #     )
            #     page = gr.Number(value=0, minimum=0, maximum=len(RESULT_DIRS), show_label=False)
            #     next_page = gr.Button(
            #         "next", variant="primary"
            #     )
            #     dir_name = gr.Markdown(value="# " + RESULT_DIRS[int(page.value)])
            #
            # with gr.Row():
            #     save_button = gr.Button("Save image", variant="primary")

                # choose_img = gr.Button("выбрать", variant="primary")
    #
    #     with gr.TabItem("Hide 2"):
    #         with gr.Row():
    #             # real_img = gr.Image(height=800, show_download_button=True)
    #             image = gr.Image(height=800, show_download_button=True)
    #             with gr.Column():
    #                 # result = gr.Dataframe(
    #                 #     headers=["text", "top", "left", "height"],
    #                 #     datatype=["str", "number", "number", "number"]
    #                 # )
    #                 #
    #                 # up_font = gr.Dropdown(
    #                 #     os.listdir(FONT_PATH), label="Верхний шрифт", value=os.listdir(FONT_PATH)[0]
    #                 # )
    #                 # down_font = gr.Dropdown(
    #                 #     os.listdir(FONT_PATH), label="Down шрифт", value=os.listdir(FONT_PATH)[1]
    #                 # )
    #                 #
    #                 # threshold = gr.Number(value=195, minimum=150, maximum=250, label="threshold")
    #                 # thickness = gr.Number(value=3, minimum=0, maximum=15, label="thickness")
    #                 #
    #                 # add_text_to_image = gr.Button("add text", variant="primary")
    #                 # add_text_from_font = gr.Button("add text own font", variant="primary")
    #                 # save_button = gr.Button("Save image", variant="primary")
    #                 pass
    #
    #     with gr.TabItem("Hide"):
    #         image_orig = gr.Image(height=800, show_download_button=True)

    start.click(
        get_imgs,
        [],
        [gallery, pathes]
    )

    page.submit(
        change_dir,
        [page],
        [page, text]
    )

    prev_page.click(
        lambda page: change_dir(page, "-"),
        [page],
        [page, text]
    )

    next_page.click(
        lambda page: change_dir(page, "+"),
        [page],
        [page, text]
    )

    gallery.select(
        select_image,
        [gallery, page],
        [image, index_path]
    )

    #
    # # add_text_to_image.click(
    # #     add_text,
    # #     [image_orig, page, thickness, threshold],
    # #     [image]
    # # )
    #
    # save_button.click(
    #     save_img,
    #     [image_orig, page],
    #     []
    # )

    # add_text_from_font.click(
    #     add_text_font,
    #     [image_orig, result, up_font, down_font],
    #     [image]
    # )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5010)
