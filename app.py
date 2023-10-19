from utils import *


with gr.Blocks() as demo:
    ocr_res = gr.State(None)
    predict_ocr = gr.State(None)
    bboxes = gr.State(None)
    map_bboxes = gr.State(None)

    history = gr.State(None)

    with gr.Row():
        inpaint_model_name = gr.Dropdown(
            label="Inpaint модель", choices=INP_MODELS, show_label=True, value=INP_MODELS[0]
        )

        model_name = gr.Dropdown(
            label="Upgrade модель", choices=MODELS, show_label=True, value=MODELS[0]
        )

        vae_name = gr.Dropdown(
            label="VAE модель", choices=VAES, show_label=True, value=VAES[0]
        )

    with gr.Tabs():
        with gr.TabItem("Remove Text"):
            with gr.Row():
                with gr.Column():
                    texted_img = gr.Image(show_download_button=True)
                    without_text = gr.Gallery(
                        object_fit="contain", label="Generated images",
                        show_label=False, elem_id="gallery", columns=5
                    )

                with gr.Column():
                    prompt_rt = gr.Textbox(label="Prompt", value="")
                    negative_prompt_rt = gr.Textbox(label="Negative Prompt", value=BASE_NEGATIV_PROMPT)
                    sampler_rt = gr.Dropdown(
                        SAMPLERS, label="Sampler", value="Euler"
                    )
                    steps_rt = gr.Slider(minimum=1, maximum=100, value=40, label="steps", show_label=True)
                    cfg_scale_rt = gr.Slider(minimum=1, maximum=20, value=7, label="cfg scale", show_label=True)
                    denoising_strength_rt = gr.Slider(
                        minimum=0.01, maximum=1, value=0.75, label="Denoising strength", show_label=True
                    )

                    frame_around_size = gr.Slider(
                        minimum=1, maximum=50, value=10, label="frame around size", show_label=True
                    )
                    batch_size_rt = gr.Number(label="batch size", value=1)

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
                        inpainted_image = gr.Gallery(
                            object_fit="contain", label="Generated images",
                            show_label=False, elem_id="gallery", columns=5
                        )
                    with gr.Column():
                        gr.Markdown("# Outpaint")
                        outpainted_image = gr.Image(show_download_button=True)
                        result_image = gr.Gallery(
                            object_fit="contain", label="Generated images",
                            show_label=False, elem_id="gallery", columns=5
                        )

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
                    batch_size = gr.Number(label="batch size", value=1)
                    generate_img = gr.Button("Сгенерировать")

                    size_choose = gr.Dropdown(
                        list(SIZES.keys()), label="Размер", value=list(SIZES.keys())[0]
                    )

                    gr.Markdown("Свои значения")
                    with gr.Row():
                        x_value = gr.Number(label="Высота", value=409)
                        y_value = gr.Number(label="Ширина", value=544)

                    with gr.Row():
                        left = gr.Checkbox(label="left")
                        right = gr.Checkbox(label="right")
                        top = gr.Checkbox(label="top")
                        bottom = gr.Checkbox(label="bottom")

                    outpaint_image = gr.Button("Дорисовать")
                    outpaint_with_value = gr.Button("Дорисовать со своими значениями")

        with gr.TabItem("Upgrade"):
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("Canny"):
                        with gr.Column():
                            with gr.Row():
                                canny_input_img = gr.Image(show_download_button=True, height=600)
                                canny_preview_img = gr.Image()
                            with gr.Row():
                                canny_low_threshold = gr.Slider(
                                    minimum=1, maximum=240, value=100, label="canny low threshold", show_label=True
                                )
                                canny_high_threshold = gr.Slider(
                                    minimum=40, maximum=240, value=200, label="canny high threshold", show_label=True
                                )
                            with gr.Row():
                                canny_preview_button = gr.Button("preview canny")
                                canny_push = gr.Button("Сгенерировать", variant="primary")
                            canny_out_img = gr.Image()

                    with gr.TabItem("Depth"):
                        with gr.Column():
                            with gr.Row():
                                depth_input_img = gr.Image(show_download_button=True, height=600)
                                depth_preview_img = gr.Image()
                            with gr.Row():
                                depth_type = gr.Dropdown(
                                    label="Depth type", choices=[
                                        "depth_leres", "depth_leres++", "depth_midas", "depth_zoe"
                                    ],
                                    show_label=True, value="depth_midas"
                                )
                                depth_near = gr.Slider(
                                    minimum=0, maximum=100, value=0, label="Remove Near %", show_label=True
                                )
                                depth_back = gr.Slider(
                                    minimum=0, maximum=100, value=0, label="Remove Background %", show_label=True
                                )
                            with gr.Row():
                                depth_preview_button = gr.Button("preview depth")
                                depth_push = gr.Button("Сгенерировать", variant="primary")
                            depth_out_img = gr.Image()

                with gr.Column():
                    prompt_cn = gr.Textbox(label="Prompt", value="")
                    negative_prompt_cn = gr.Textbox(label="Negative Prompt", value=BASE_NEGATIV_PROMPT)
                    sampler_cn = gr.Dropdown(
                        SAMPLERS, label="Sampler", value="Euler"
                    )
                    steps_cn = gr.Slider(minimum=1, maximum=100, value=40, label="steps", show_label=True)
                    cfg_scale_cn = gr.Slider(minimum=1, maximum=20, value=7, label="cfg scale", show_label=True)
                    denoising_strength_cn = gr.Slider(
                        minimum=0.01, maximum=1, value=0.75, label="Denoising strength", show_label=True
                    )
                    guidance_start = gr.Slider(
                        minimum=0, maximum=0.9, value=0, label="Guidance start", show_label=True
                    )
                    guidance_end = gr.Slider(
                        minimum=0.1, maximum=1, value=1, label="Guidance end", show_label=True
                    )

                    control_mode = gr.Dropdown(
                        label="Control Mode", choices=CONTROL_MODE, show_label=True, value=CONTROL_MODE[0]
                    )

        with gr.TabItem("hide"):
            image_orig = gr.Image(height=800, show_download_button=True)
            original_img = gr.Image(show_download_button=False)

    texted_img.upload(
        ocr_detect,
        [texted_img],
        [texted_img, image_orig, bboxes, ocr_res, map_bboxes]
    )

    texted_img.select(
        choose_bboxes,
        [texted_img, bboxes, map_bboxes],
        [texted_img, bboxes, map_bboxes]
    )

    remove_text_but.click(
        remove_text,
        [
            image_orig, bboxes, map_bboxes, prompt_rt, negative_prompt_rt, inpaint_model_name, vae_name,
            sampler_rt, steps_rt, cfg_scale_rt, denoising_strength_rt, frame_around_size, batch_size_rt
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
        [
            masked_image, inpaint_model_name, vae_name, prompt, negative_prompt,
            sampler, steps, cfg_scale, denoising_strength, batch_size
        ],
        [inpainted_image]
    )

    outpaint_image.click(
        outpainting_with_value,
        [
            outpainted_image, inpaint_model_name, vae_name, left, top, right, bottom, size_choose,
            prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
        ],
        [result_image]
    )

    outpaint_with_value.click(
        outpainting,
        [
            outpainted_image, inpaint_model_name, vae_name, left, top, right, bottom, x_value, y_value,
            prompt, negative_prompt, sampler, steps, cfg_scale, denoising_strength, batch_size
        ],
        [result_image]
    )

    canny_preview_button.click(
        canny_preview,
        [canny_input_img, canny_low_threshold, canny_high_threshold],
        [canny_preview_img]
    )

    depth_preview_button.click(
        depth_preview,
        [depth_input_img, depth_type, depth_near, depth_back],
        [depth_preview_img]
    )

    canny_push.click(
        canny_generate,
        [
            canny_input_img, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn, cfg_scale_cn,
            denoising_strength_cn, guidance_start, guidance_end, control_mode, canny_low_threshold, canny_high_threshold
        ],
        [canny_out_img]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5000)
