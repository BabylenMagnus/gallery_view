from utils import *


with gr.Blocks() as demo:
    ocr_res = gr.State(None)
    predict_ocr = gr.State(None)
    bboxes = gr.State(None)
    map_bboxes = gr.State(None)

    model_name = gr.Dropdown(
        label="Выберите модель", choices=MODELS, show_label=True, value=MODELS[0]
    )

    with gr.Tabs():
        with gr.TabItem("Remove Text"):
            with gr.Row():
                with gr.Column():
                    texted_img = gr.Image(show_download_button=True)
                    without_text = gr.Image(label='Without text')
                    with_text = gr.Image(label='With text')

                with gr.Column():
                    font_choose = gr.Dropdown(
                        os.listdir(FONT_PATH), label="Шрифт", value=os.listdir(FONT_PATH)[0]
                    )
                    color = gr.Textbox(value="white")
                    text_table = gr.Dataframe(
                        headers=["text", "top", "height", "left"],
                        datatype=["str", "number", "number", "number"]
                    )
                    remove_text_but = gr.Button("Remove Text", variant="primary")
                    add_text = gr.Button("Add Text")

        with gr.TabItem("Inpaint"):
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("# Inpaint")
                        masked_image = gr.Image(
                            interactive=True, tool='sketch', elem_id="image_upload", mask_opacity=1
                        )
                        inpainted_image = gr.Image(label="Результат")
                    with gr.Column():
                        gr.Markdown("# Outpaint")
                        outpainted_image = gr.Image(show_download_button=True)
                        result_image = gr.Image()

                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="")
                    negative_prompt = gr.Textbox(label="Negative Prompt", value=BASE_NEGATIV_PROMPT)
                    sampler = gr.Dropdown(
                        SAMPLERS, label="Sampler", value="Euler"
                    )
                    steps = gr.Slider(minimum=1, maximum=100, value=40, label="steps", show_label=True)
                    cfg_scale = gr.Slider(minimum=1, maximum=20, value=7, label="cfg scale", show_label=True)
                    denoising_strength = gr.Slider(
                        minimum=0.01, maximum=1, value=0.75, label="Denoising strength", show_label=True
                    )

                    generate_img = gr.Button("Сгенерировать")

                    size_choose = gr.Dropdown(
                        list(SIZES.keys()), label="Размер", value=list(SIZES.keys())[0]
                    )

                    with gr.Row():
                        left = gr.Checkbox(label="left")
                        right = gr.Checkbox(label="right")
                        top = gr.Checkbox(label="top")
                        bottom = gr.Checkbox(label="bottom")

                    outpaint_image = gr.Button("Дорисовать")

        with gr.TabItem("Hide"):
            image_orig = gr.Image(height=800, show_download_button=True)

    texted_img.upload(
        ocr_detect,
        [texted_img],
        [texted_img, image_orig, bboxes, ocr_res, predict_ocr]
    )

    texted_img.select(
        choose_bboxes,
        [texted_img, bboxes, map_bboxes],
        [texted_img, bboxes, map_bboxes]
    )

    remove_text_but.click(
        remove_text,
        [
            texted_img, bboxes, map_bboxes, predict_ocr, prompt,
            negative_prompt, model_name, sampler, steps, cfg_scale, denoising_strength
        ],
        [without_text, text_table]
    )

    add_text.click(
        add_text_font,
        [without_text, text_table, font_choose, color],
        [with_text]
    )

    generate_img.click(
        inpaint_image,
        [masked_image, model_name, prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength],
        [inpainted_image]
    )

    outpaint_image.click(
        outpainting,
        [
            outpainted_image, model_name, left, top, right, bottom, size_choose,
            prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength
        ],
        [result_image]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5000)
