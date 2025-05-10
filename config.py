class Config:
    # 镜像源配置
    HF_ENDPOINT = "https://hf-mirror.com"

    # 模型配置
    BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    DEVICE = "cuda"   
    TORCH_DTYPE = "float16"    #半精度加速

    # 文案生成模板
    PROMPT_TEMPLATES = {
        "小红书": (
            "你是一个社交媒体专家，请为以下图片生成一条适合小红书的帖子：\n"
            "- 包含1-2个emoji\n"
            "- 添加3个相关话题标签(以#开头)\n"
            "- 使用口语化的中文\n"
            "- 结尾提出一个问题\n"
            "图片内容: {description}"
        ),
        "Instagram": (
            "Generate an engaging Instagram post in English with:\n"
            "- 2-3 emojis\n"
            "- 3 hashtags\n"
            "- A question to encourage interaction\n"
            "Image content: {description}"
        )
    }
