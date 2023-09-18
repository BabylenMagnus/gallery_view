import gradio as gr
from utils import *


FONT_PATH = "fonts/"
MODELS = get_models()


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
                    prompt_text = gr.Textbox(label="Prompt", value="")
                    negative_prompt_text = gr.Textbox(label="Negative Prompt", value=BASE_NEGATIV_PROMPT)

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
                    generate_img = gr.Button("Сгенерировать")

                    size_choose = gr.Dropdown(
                        list(SIZES.keys()), label="Размер", value=list(SIZES.keys())[0]
                    )

                    with gr.Row():
                        left = gr.Checkbox(label="left")
                        right = gr.Checkbox(label="right")
                        top = gr.Checkbox(label="top")
                        bottom = gr.Checkbox(label="bottom")

                    outpaint_image = gr.Button("Сгенерировать")


        # with gr.TabItem("Outpaint"):
        #     with gr.Row():
        #         with gr.Column():
        #         with gr.Column():
        #             prompt_out = gr.Textbox(label="Prompt", value="")
        #             negative_prompt_out = gr.Textbox(label="Negative Prompt", value=BASE_NEGATIV_PROMPT)
        #
        #             size_choose = gr.Dropdown(
        #                 list(SIZES.keys()), label="Размер", value=list(SIZES.keys())[0]
        #             )
        #
        #             with gr.Row():
        #                 left = gr.Checkbox(label="left")
        #                 right = gr.Checkbox(label="right")
        #                 top = gr.Checkbox(label="top")
        #                 bottom = gr.Checkbox(label="bottom")
        #
        #             outpaint_image = gr.Button("Сгенерировать")

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
        [image_orig, bboxes, map_bboxes, predict_ocr, prompt_text, negative_prompt_text, model_name],
        [without_text, text_table]
    )

    add_text.click(
        add_text_font,
        [without_text, text_table, font_choose, color],
        [with_text]
    )

    generate_img.click(
        inpaint_image,
        [masked_image, model_name, prompt, negative_prompt],
        [inpainted_image]
    )

    outpaint_image.click(
        outpainting,
        [outpainted_image, model_name, left, top, right, bottom, size_choose, prompt, negative_prompt],
        [result_image]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=5010)
