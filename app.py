import glob
import os

from utils import *


RESULT_DIR = r"IMGS_1_11"
RESULTS = os.listdir(RESULT_DIR)
OUT_DIR = r"WITHOUT_TEXT"


def get_img(num):
    return Image.open(os.path.join(RESULT_DIR, RESULTS[int(num)])).convert("RGB")


def save_imgs(imgs, page):
    for i, img in enumerate(imgs):
        img = Image.open(img["name"])
        img.save(os.path.join(
            OUT_DIR, f"{RESULTS[int(page)]}_page{len(glob.glob(OUT_DIR + f'/{RESULTS[int(page)]}_page*'))}.png"
        ))


def change_dir(num, button=None):
    if button is None:
        img = get_img(num)
        return img, num, ocr_detect(np.array(img))
    if button == "+" and num < len(RESULTS):
        num += 1
    elif num > 0:
        num -= 1
    print(RESULTS[int(num)])
    img = get_img(num)
    return img, num, img, *ocr_detect(np.array(img))


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

                    save_images_remove_text = gr.Button("Save", variant="primary")

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
                    save_images_upgrade = gr.Button("Save", variant="primary")

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
                            canny_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

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
                            depth_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

                    with gr.TabItem("NormalMap"):
                        with gr.Column():
                            with gr.Row():
                                normal_input_img = gr.Image(show_download_button=True, height=600)
                                normal_preview_img = gr.Image()
                            with gr.Row():
                                normal_type = gr.Dropdown(
                                    label="Normal type", choices=[
                                        "normal_bae", "normal_midas"
                                    ],
                                    show_label=True, value="normal_bae"
                                )
                                normal_threshold = gr.Slider(
                                    minimum=0, maximum=1, value=0.4, label="Background Threshold", show_label=True
                                )
                            with gr.Row():
                                normal_preview_button = gr.Button("preview depth")
                                normal_push = gr.Button("Сгенерировать", variant="primary")
                            normal_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

                    with gr.TabItem("OpenPose"):
                        with gr.Column():
                            with gr.Row():
                                pose_input_img = gr.Image(show_download_button=True, height=600)
                                pose_preview_img = gr.Image()
                            with gr.Row():
                                pose_type = gr.Dropdown(
                                    label="Pose type", choices=[
                                        "dw_openpose_full", "openpose", "openpose_face", "openpose_faceonly",
                                        "openpose_full", "openpose_hand"
                                    ],
                                    show_label=True, value="openpose"
                                )
                            with gr.Row():
                                pose_preview_button = gr.Button("preview depth")
                                pose_push = gr.Button("Сгенерировать", variant="primary")
                            pose_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

                    with gr.TabItem("Lineart"):
                        with gr.Column():
                            with gr.Row():
                                line_input_img = gr.Image(show_download_button=True, height=600)
                                line_preview_img = gr.Image()
                            with gr.Row():
                                line_type = gr.Dropdown(
                                    label="Lineart type", choices=[
                                        "lineart_anime", "lineart_anime_denoise", "lineart_coarse", "lineart_realistic",
                                        "lineart_standard (from white bg & black line)",
                                        "invert (from white bg & black line)"
                                    ],
                                    show_label=True, value="lineart_standard (from white bg & black line)"
                                )
                            with gr.Row():
                                line_preview_button = gr.Button("preview line")
                                line_push = gr.Button("Сгенерировать", variant="primary")
                            line_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

                    with gr.TabItem("Shuffle"):
                        with gr.Column():
                            with gr.Row():
                                shuffle_input_img = gr.Image(show_download_button=True, height=600)
                            with gr.Row():
                                shuffle_push = gr.Button("Сгенерировать", variant="primary")
                            shuffle_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

                    with gr.TabItem("Reference"):
                        with gr.Column():
                            with gr.Row():
                                reference_input_img = gr.Image(show_download_button=True, height=600)
                            with gr.Row():
                                reference_type = gr.Dropdown(
                                    label="Reference type", choices=[
                                        "reference_adain", "reference_adain+attn", "reference_only"
                                    ],
                                    show_label=True, value="reference_only"
                                )
                            with gr.Row():
                                reference_push = gr.Button("Сгенерировать", variant="primary")
                            reference_out_img = gr.Gallery(
                                object_fit="contain", label="Generated images",
                                show_label=False, elem_id="gallery", columns=5
                            )

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
                    batch_size_cn = gr.Number(label="batch size", value=1)

                    with gr.Row():
                        lora_add_detail = gr.Checkbox(label="add detail")
                        lora_add_detail_value = gr.Slider(
                            minimum=-1, maximum=2, value=1, show_label=False
                        )
                    with gr.Row():
                        lora_add_details = gr.Checkbox(label="add detail #2")
                        lora_add_details_value = gr.Slider(
                            minimum=-0.5, maximum=1.5, value=1, show_label=False
                        )
                    with gr.Row():
                        lora_blindbox = gr.Checkbox(label="3DMM")
                    with gr.Row():
                        lora_eyes_gen = gr.Checkbox(label="Eyes gen")
                        lora_eyes_gen_value = gr.Slider(
                            minimum=0.1, maximum=.5, value=0.4, show_label=False
                        )
                    with gr.Row():
                        lora_polyhedron_fem = gr.Checkbox(label="Eyes gen female")
                        lora_polyhedron_fem_value = gr.Slider(
                            minimum=0.1, maximum=.5, value=0.4, show_label=False
                        )
                    with gr.Row():
                        lora_polyhedron_man = gr.Checkbox(label="Eyes gen man")
                        lora_polyhedron_man_value = gr.Slider(
                            minimum=0.1, maximum=.5, value=0.4, show_label=False
                        )
                    with gr.Row():
                        lora_beautiful_detailed = gr.Checkbox(label="Beautiful Eyes")
                        lora_beautiful_detailed_value = gr.Slider(
                            minimum=0.1, maximum=1, value=0.5, show_label=False
                        )

        with gr.TabItem("Choose"):
            with gr.Row():
                prev_page = gr.Button(
                    "prev", variant="primary"
                )
                page = gr.Number(value=0, minimum=0, maximum=len(RESULTS), show_label=False)
                next_page = gr.Button(
                    "next", variant="primary"
                )
            choose_show_img = gr.Image(show_download_button=False)

        with gr.TabItem("hide"):
            image_orig = gr.Image(height=800, show_download_button=True)
            original_img = gr.Image(show_download_button=False)
            test_img = gr.Image(show_download_button=False)

    prev_page.click(
        lambda page: change_dir(page, "-"),
        [page],
        [choose_show_img, page, outpainted_image, texted_img, image_orig, bboxes, ocr_res, map_bboxes]
    )

    next_page.click(
        lambda page: change_dir(page, "+"),
        [page],
        [choose_show_img, page, outpainted_image, texted_img, image_orig, bboxes, ocr_res, map_bboxes]
    )

    save_images_remove_text.click(
        save_imgs,
        [without_text, page],
        []
    )

    save_images_upgrade.click(
        save_imgs,
        [result_image, page],
        []
    )

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

    normal_preview_button.click(
        normal_preview,
        [normal_input_img, normal_type, normal_threshold],
        [normal_preview_img]
    )

    pose_preview_button.click(
        pose_preview,
        [pose_input_img, pose_type],
        [pose_preview_img]
    )

    line_preview_button.click(
        pose_preview,
        [line_input_img, line_type],
        [line_preview_img]
    )

    canny_push.click(
        canny_generate,
        [
            canny_input_img, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn, cfg_scale_cn,
            denoising_strength_cn, batch_size_cn, guidance_start, guidance_end, control_mode, lora_add_detail,
            lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox, lora_eyes_gen,
            lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value,  lora_polyhedron_man,
            lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, canny_low_threshold,
            canny_high_threshold
        ],
        [canny_out_img, canny_preview_img]
    )

    depth_push.click(
        depth_generate,
        [
            depth_input_img, depth_type, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn,
            cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end, control_mode,
            lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox,
            lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
            lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, depth_near, depth_back
        ],
        [depth_out_img, depth_preview_img]
    )

    normal_push.click(
        normal_generate,
        [
            normal_input_img, normal_type, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn,
            cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end, control_mode,
            lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox,
            lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value, lora_polyhedron_man,
            lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value, normal_threshold
        ],
        [normal_out_img, normal_preview_img]
    )

    pose_push.click(
        pose_generate,
        [
            pose_input_img, pose_type, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn,
            cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end, control_mode,
            lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox,
            lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value,
            lora_polyhedron_man, lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
        ],
        [pose_out_img, pose_preview_img]
    )

    line_push.click(
        line_generate,
        [
            line_input_img, line_type, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn,
            cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end, control_mode,
            lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value, lora_blindbox,
            lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value,  lora_polyhedron_man,
            lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
        ],
        [line_out_img, line_preview_img]
    )

    shuffle_push.click(
        shuffle_generate,
        [
            shuffle_input_img, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn, steps_cn,
            cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end,
            control_mode, lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value,
            lora_blindbox, lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value,
            lora_polyhedron_man, lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
        ],
        [shuffle_out_img, test_img]
    )

    reference_push.click(
        reference_generate,
        [
            reference_input_img, reference_type, model_name, vae_name, prompt_cn, negative_prompt_cn, sampler_cn,
            steps_cn, cfg_scale_cn, denoising_strength_cn, batch_size_cn, guidance_start, guidance_end,
            control_mode, lora_add_detail, lora_add_detail_value, lora_add_details, lora_add_details_value,
            lora_blindbox, lora_eyes_gen, lora_eyes_gen_value, lora_polyhedron_fem, lora_polyhedron_fem_value,
            lora_polyhedron_man, lora_polyhedron_man_value, lora_beautiful_detailed, lora_beautiful_detailed_value
        ],
        [reference_out_img, test_img]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7000)
