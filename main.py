import os
import numpy as np
import json

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import glob
import pandas as pd
import textwrap


RESULT_PATH = "/home/jjjj/Documents/gallery_view/IMGS_22.11/V1R/"
save_path = "saves/"
log_path = "log.txt"
INPUT_CSV = f"Game Icons 15.09 - with_image.csv"
FONT_PATH = "fonts"

ORG_IMG = "ALL_IMGS/"

RESULT_DIRS = [i for i in os.listdir(RESULT_PATH) if os.path.isdir(os.path.join(RESULT_PATH, i))]

with open("bboxes.json", "r") as t:
    BBOXES = json.load(t)

name_provider = {}
xl = pd.read_csv(INPUT_CSV)

for i in range(len(xl)):
    i = xl.loc[i]
    if not isinstance(i['name'], str):
        continue
    if i['provider'] in ["pragmaticplaylive", "hogaming", "xprogaming", "ezugi", "vivogaming"]:
        continue
    name = i['name'].strip().lower().replace('`', "'")

    if i['provider'] == 'relaxgamingslots':
        name = name.replace('mobile', '').strip()

    name_provider[name] = i['provider']


def get_imgs(num):
    return [Image.open(i) for i in glob.glob(RESULT_PATH + RESULT_DIRS[int(num)] + "/*.png")]


def change_dir(num, button=None):
    if button is None:
        return get_imgs(num), "# " + RESULT_DIRS[int(num)], Image.open(
        os.path.join(ORG_IMG, RESULT_DIRS[int(num)] + ".png")
    )
    if button == "+" and num < len(RESULT_DIRS):
        num += 1
    elif num > 0:
        num -= 1
    return get_imgs(num), num, "# " + RESULT_DIRS[int(num)], Image.open(
        os.path.join(ORG_IMG, RESULT_DIRS[int(num)] + ".png")
    )


def select_image(evt: gr.SelectData, gallery, name):
    name = RESULT_DIRS[int(name)]
    provider = name_provider[name].upper()
    bboxes = BBOXES[name]
    bboxes = sorted(bboxes, key=lambda x: x[0][1])

    name = name.upper()

    l = len(name)
    n = len(bboxes) - 1
    out = textwrap.wrap(name, l)
    while len(out) < n or min(map(len, out)) == 1:
        l -= 1
        out = textwrap.wrap(name, l)

    out_font = []
    out_start = []
    for i in bboxes:
        j = [j[1] for j in i]
        out_font.append(int((max(j) - min(j)) * 0.65))
        out_start.append(min(j) - 10)

    out = out + ['' for _ in range(3 - len(out))]
    out_font = out_font + [10 for _ in range(3 - len(out_font))]
    out_start = out_start + [400 for _ in range(3 - len(out_start))]

    i = Image.open(gallery.root[int(evt.index)].image.path)

    return i, i, *out, *out_font, *out_start, provider


def add_text_font(
        real_img, fst_txt, scn_txt, thr_txt, prv_txt, fst_fnt, scn_fnt, thr_fnt, prv_fnt,
        fst_start, scn_start, thr_start, prv_start
):
    img = Image.fromarray(real_img.copy())

    for txt, fnt, start in zip(
            [fst_txt, scn_txt, thr_txt],
            [fst_fnt, scn_fnt, thr_fnt],
            [fst_start, scn_start, thr_start]
    ):
        image = Image.new('RGB', img.size, "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(os.path.join(FONT_PATH, "RubikWetPaint-Regular.ttf"), fnt)

        _, _, w, h = draw.textbbox(
            (0, 0), txt, font=font
        )

        draw = ImageDraw.Draw(img)
        draw.text(((img.size[0] - w) / 2, int(start)), txt, font=font, fill="white")

    image = Image.new('RGB', img.size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(os.path.join(FONT_PATH, "RedHatText-VariableFont_wght.ttf"), prv_fnt)

    _, _, w, h = draw.textbbox(
        (0, 0), prv_txt, font=font
    )

    draw = ImageDraw.Draw(img)
    draw.text(((img.size[0] - w) / 2, int(prv_start)), prv_txt, font=font, fill="white")

    return img


def load_img_to_array(img_p):
    img = Image.open(img_p)
    img = img.convert("RGB")
    return np.array(img)


def save_img(img, page):
    img = Image.fromarray(img)
    name = RESULT_DIRS[int(page)]
    print("save ", name)
    img.save(save_path + name + str(len(glob.glob(save_path + name + "*.png"))) + ".png")
    return img


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
                label="Generated images", show_label=False, elem_id="gallery", columns=7, height=800
            )
            with gr.Row():
                org_img = gr.Image(width=300)
                choosed_img = gr.Image(width=300)

        with gr.TabItem("Add text"):
            with gr.Row():
                real_img = gr.Image(height=600)
                texted_image = gr.Image(height=600)
                with gr.Column():

                    with gr.Row():
                        fst_txt = gr.Textbox(value="")
                        fst_fnt = gr.Slider(minimum=1, maximum=120, value=60)
                        fst_start = gr.Slider(minimum=300, maximum=540, value=320)

                    with gr.Row():
                        scn_txt = gr.Textbox(value="")
                        scn_fnt = gr.Slider(minimum=10, maximum=120, value=60)
                        scn_start = gr.Slider(minimum=300, maximum=540, value=320)

                    with gr.Row():
                        thr_txt = gr.Textbox(value="")
                        thr_fnt = gr.Slider(minimum=10, maximum=120, value=60)
                        thr_start = gr.Slider(minimum=300, maximum=540, value=320)

                    with gr.Row():
                        prv_txt = gr.Textbox(value="")
                        prv_fnt = gr.Slider(minimum=10, maximum=120, value=34)
                        prv_start = gr.Slider(minimum=300, maximum=540, value=490)

                    add_text_from_font = gr.Button("Take Text Font", variant="primary")
                    save_button = gr.Button("Save image", variant="primary")
            last_save = gr.Image(width=300)

    page.submit(
        change_dir,
        [page],
        [gallery, dir_name, org_img]
    )

    prev_page.click(
        lambda page: change_dir(page, "-"),
        [page],
        [gallery, page, dir_name, org_img]
    )

    next_page.click(
        lambda page: change_dir(page, "+"),
        [page],
        [gallery, page, dir_name, org_img]
    )

    gallery.select(
        select_image,
        [gallery, page],
        [
            choosed_img, real_img,
            fst_txt, scn_txt, thr_txt,
            fst_fnt, scn_fnt, thr_fnt,
            fst_start, scn_start, thr_start,
            prv_txt
        ]
    )

    add_text_from_font.click(
        add_text_font,
        [
            real_img,
            fst_txt, scn_txt, thr_txt, prv_txt,
            fst_fnt, scn_fnt, thr_fnt, prv_fnt,
            fst_start, scn_start, thr_start, prv_start
        ],
        [texted_image]
    )

    save_button.click(
        save_img,
        [texted_image, page],
        [last_save]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5010)
