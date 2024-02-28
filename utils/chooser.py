import os
import numpy as np
import glob

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "fonts"


def draw_textbox(draw, txt, font, img, start):
    _, _, w, h = draw.textbbox(
        (0, 0), txt, font=font
    )

    draw = ImageDraw.Draw(img)
    draw.text(((img.size[0] - w) / 2, int(start)), txt, font=font, fill="white")


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
        draw_textbox(draw, txt, font, img, start)

    image = Image.new('RGB', img.size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(os.path.join(FONT_PATH, "RedHatText-VariableFont_wght.ttf"), prv_fnt)
    draw_textbox(draw, prv_txt, font, img, prv_start)

    return img


def load_img_to_array(img_p):
    img = Image.open(img_p)
    img = img.convert("RGB")
    return np.array(img)
