import os
import time

import gradio as gr
from PIL import Image
import glob
import pandas as pd

from utils.draw_text import draw_casual_text


RESULT_PATH = r"OUTPUT\2_23_OUTPAINT\\"
save_path = r"OUTPUT\2_26_saves\\"
INPUT_CSV = f"data/2_21_without_image.csv"

RESULT_DIRS = [i for i in os.listdir(RESULT_PATH) if os.path.isdir(os.path.join(RESULT_PATH, i))]

HASH2PROVIDER = {}
HASH2NAME = {}

xl = pd.read_csv(INPUT_CSV)

for i in range(len(xl)):
    i = xl.iloc[i]

    if not isinstance(i['Name'], str):
        continue

    name = i['Hash']
    HASH2PROVIDER[name] = i['Provider']
    HASH2NAME[name] = i["Name"]


def get_imgs(num):
    start = time.time()
    a = [Image.open(i) for i in glob.glob(RESULT_PATH + RESULT_DIRS[int(num)] + "/*.png")]
    print(time.time() - start)
    return a


def change_dir(num, button=None):
    if button is None:
        return get_imgs(num), "# " + HASH2NAME[RESULT_DIRS[int(num)]]
    if button == "+" and num < len(RESULT_DIRS):
        num += 1
    elif num > 0:
        num -= 1
    return get_imgs(num), num, "# " + HASH2NAME[RESULT_DIRS[int(num)]]


def select_image(evt: gr.SelectData, gallery, name):
    # name = RESULT_DIRS[int(name)]
    # slot_name = HASH2NAME[name].upper()
    # provider = HASH2PROVIDER[name].upper()
    img = Image.open(gallery[int(evt.index)]['name'])

    return img, img


def add_text(choosed_img, page):
    name = RESULT_DIRS[int(page)]
    slot_name = HASH2NAME[name].upper()
    provider = HASH2PROVIDER[name].upper()

    return draw_casual_text(Image.fromarray(choosed_img), slot_name, provider)


def save_image(image, page):
    img = Image.fromarray(image)
    name = RESULT_DIRS[int(page)]
    print("save ", HASH2NAME[name])
    if os.path.exists(save_path + name + ".png"):
        img.save(save_path + name + "_" + str(len(glob.glob(save_path + name + "*.png"))) + ".png")
    else:
        img.save(save_path + name + ".png")
    return img


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Main"):
            with gr.Row():
                prev_page = gr.Button(
                    "prev", variant="primary"
                )
                page = gr.Number(value=0, minimum=0, maximum=len(RESULT_DIRS), show_label=False)
                next_page = gr.Button(
                    "next", variant="primary"
                )
                dir_name = gr.Markdown(value="# " + HASH2NAME[RESULT_DIRS[int(page.value)]])

            gallery = gr.Gallery(
                value=get_imgs(page.value), object_fit="contain",
                label="Generated images", show_label=False, elem_id="gallery", columns=7, height=800
            )
            with gr.Row():
                choosed_img = gr.Image(width=400)
                last_saved_img = gr.Image(width=400)
                with gr.Column():
                    add_text_button = gr.Button("Add Text")
                    save_button = gr.Button("Save")

        with gr.TabItem("Hide"):
            orig_image = gr.Image(width=400)

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
        [gallery, page],
        [choosed_img, orig_image]
    )

    add_text_button.click(
        add_text,
        [orig_image, page],
        [choosed_img]
    )

    save_button.click(
        save_image,
        [choosed_img, page],
        [last_saved_img]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5010)
