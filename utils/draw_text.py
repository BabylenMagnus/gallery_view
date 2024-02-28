from PIL import Image, ImageDraw, ImageFont


splits = {
    1: [[340, 78]],
    2: [[355, 56], [400, 68]],
    3: [[310, 68], [370, 44], [420, 68]],
    4: [[300, 68], [350, 58], [390, 68], [420, 68]]
}

UP_FONT_PATH = "fonts/RubikWetPaint-Regular.ttf"

FONT_PATH = "fonts/RedHatText-VariableFont_wght.ttf"

TOP_PROVIDER = 510
TOPS = [310, 350, 350, 340, 340]


def split_text(text):
    text = text.replace("-", "").upper()
    text = text.split()

    THRESH = 9
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
            if i == "THE":
                new_text.append([a.strip()])
                new_text.append([i.strip()])
                a = ""
                continue
            if len(a) < THRESH:
                a += " " + i
            else:
                new_text.append([a.strip()])
                a = i
        new_text.append([a.strip()])
    return new_text


def draw_casual_text(img, text, provider):
    text = split_text(text)
    img = img.convert("RGB")

    s = splits[len(text)]
    top = TOPS[len(text)]

    for t in text:
        t = " ".join(t)
        image = Image.new('RGB', img.size, "white")
        draw = ImageDraw.Draw(image)

        for f_size in [38, 42, 46, 48, 52, 56, 64, 72, 80, 98, 106, 112]:
            font = ImageFont.truetype(UP_FONT_PATH, f_size)

            _, _, w, h = draw.textbbox(
                (0, 0), t, font=font
            )
            if w > 360 or h > (150 / len(text)):
                draw = ImageDraw.Draw(img)
                draw.text(((img.size[0] - w) / 2, top), t, font=font, fill="white")
                top += h - 10
                break

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 34)

    _, _, w, h = draw.textbbox(
        (0, 0), provider, font=font
    )

    draw = ImageDraw.Draw(img)
    draw.text(((img.size[0] - w) / 2, TOP_PROVIDER), provider, font=font, fill="white")

    return img
