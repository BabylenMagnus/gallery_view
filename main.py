import os
import numpy as np

import gradio as gr
from PIL import Image
import glob


RESULT_PATH = "IMGS/"
save_path = "saves/"
log_path = "log.txt"

RESULT_DIRS = [i for i in os.listdir(RESULT_PATH) if os.path.isdir(os.path.join(RESULT_PATH, i))]


def get_imgs(num):
    return [Image.open(i) for i in glob.glob(RESULT_PATH + RESULT_DIRS[int(num)] + "/*.png")]


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
                label="Generated images", show_label=False, elem_id="gallery", columns=7, height=650
            )
            save_button = gr.Button("Save image", variant="primary")
            with gr.Row():
                choosed_img = gr.Image(width=300)
                last_save = gr.Image(width=300)

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
        [choosed_img]
    )

    save_button.click(
        save_img,
        [choosed_img, page],
        [last_save]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5010)
