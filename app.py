from utils import *


with gr.Blocks() as demo:
    ocr_res = gr.State(None)
    predict_ocr = gr.State(None)
    bboxes = gr.State(None)
    map_bboxes = gr.State(None)

    history = gr.State(None)

    model_name = gr.Dropdown(
        label="Выберите модель", choices=MODELS, show_label=True, value=MODELS[0]
    )

    with gr.Tabs():
        with gr.TabItem("Remove Text"):
            with gr.Row():
                with gr.Column():
                    texted_img = gr.Image(show_download_button=True)
                    without_text = gr.Image(label='Without text')

                with gr.Column():
                    remove_text_but = gr.Button("Remove Text", variant="primary")

        with gr.TabItem("Add Text"):
            with gr.Row():
                input_img = gr.Image(label="Input")
                output_img = gr.Image(label="Modified Image")
                with gr.Column():
                    font_choose = gr.Dropdown(
                        os.listdir(FONT_PATH), label="Шрифт", value=os.listdir(FONT_PATH)[0]
                    )
                    font_size = gr.Number(label="Размер шрифта", value=24)
                    color = gr.Textbox(label="Цвет", value="white")

                    text_input = gr.Textbox(label="Текст")

                    with gr.Row():
                        x_cord = gr.Slider(minimum=1, maximum=1000, value=40, label="X", show_label=True)
                        y_cord = gr.Slider(minimum=1, maximum=1000, value=40, label="Y", show_label=True)

                    add_text_button = gr.Button("Добавить текст по координатам")
                    save_button = gr.Button("Сохранить текст", variant="primary")
                    undo_button = gr.Button("Отменить")

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

        with gr.TabItem("hide"):
            image_orig = gr.Image(height=800, show_download_button=True)
            original_img = gr.Image(show_download_button=False)

    texted_img.upload(
        ocr_detect,
        [texted_img],
        [texted_img, image_orig, bboxes, ocr_res]
    )

    texted_img.select(
        choose_bboxes,
        [texted_img, bboxes, map_bboxes],
        [texted_img, bboxes, map_bboxes]
    )

    remove_text_but.click(
        remove_text,
        [
            texted_img, bboxes, map_bboxes, prompt, negative_prompt, model_name, sampler, steps,
            cfg_scale, denoising_strength
        ],
        [without_text]
    )

    input_img.upload(
        lambda x: (x, x),
        [input_img],
        [input_img, original_img]
    )

    input_img.select(
        add_text_to_coords,
        [input_img, text_input, font_choose, font_size, color],
        [output_img, x_cord, y_cord]
    )

    add_text_button.click(
        add_text_to_xy,
        [input_img, text_input, x_cord, y_cord, font_choose, font_size, color],
        output_img
    )

    save_button.click(
        save_text,
        [input_img, text_input, x_cord, y_cord, font_choose, font_size, color, history],
        [input_img, history]
    )

    undo_button.click(
        undo,
        [history, original_img],
        [history, input_img]
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
