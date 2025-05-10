import gradio as gr
from PIL import Image
from config import Config
from utils.model_loader import load_blip2, load_llava
import os


# 设置镜像环境变量
os.environ["HF-ENDPOINT"] = Config.HF_ENDPOINT

# 预加载模型
blip_processor, blip_model = load_blip2()
llava_processor, llava_model = load_llava()

def generate_description(image: Image.Image) -> str:
    """生成基础图像描述"""
    inputs = blip_processor(
        image=image,
        text="Question: 请详细描述这张图片的内容 Answer:",
        return_tensors="pt",
    ).to(blip_model.device)

    generated_ids = blip_model.generate(**inputs, max_new_tokens=100)
    description = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return description.strip("Answer:")[-1].strip()

def generate_post(image_path: str, platform: str) -> str:
    """生成平台定制化文案"""
    try:
        image = Image.open(image_path).convert("RGB")

        # 第一步：生成基础描述
        description = generate_description(image)
        print(f"[DEBUG] 图像描述: {description}")

        # 第二步：根据平台生成文案
        prompt_template = Config.PROMPT_TEMPLATES.get(platform, Config.PROMPT_TEMPLATES["小红书"])
        prompt = prompt_template.format(description=description)

        inputs = llava_processor(
            text=prompt,
            image=image,
            return_tensors="pt",
        ).to(llava_model.device)

        outputs = llava_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        full_response = llava_processor.decode(outputs[0], skip_special_tokens=True)

        # 提取ASSISTANT后的内容
        return full_response.split("ASSISTANT:")[-1].strip()

    except Exception as e:
        return f"生成失败: {str(e)}"

    # 构建Gradio界面
with gr.Blocks(title="智能文案生成器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 智能社交媒体文案生成器")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="上传图片", height=300)
            platform_dropdowm = gr.Dropdown(
                choices=["小红书", "Instagram"],
                value="小红书",
                label="选择平台"
            )
            generate_btn = gr.Button("开始生成", variant="primary")

        output_text = gr.Textbox(
            label="生成结果",
            placeholder="文案将在这里显示...",
            lines=8,
            interactive=False
        )

    # 事件绑定
    generate_btn.click(
        fn=generate_post,
        inputs=[image_input, platform_dropdowm],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
