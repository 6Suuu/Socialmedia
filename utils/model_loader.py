from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration
from config import Config
import torch

def load_blip2():
    """加载BLIP-2图像描述模型"""
    processor = Blip2Processor.from_pretrained(
        Config.BLIP2_MODEL_NAME,
        use_fast=False,
        cache_dir="./models"
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.BLIP2_MODEL_NAME,
        torch_dtype=getattr(torch, Config.TORCH_DTYPE),
        device_map=Config.DEVICE,
        cache_dir="./models"
    )
    return processor, model

def load_llava():
    """加载LLaVA多模态生成模型"""
    processor = AutoProcessor.from_pretrained(
        Config.LLAVA_MODEL_NAME,
        cache_dir="./models"
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        Config.LLAVA_MODEL_NAME,
        torch_dtype=getattr(torch, Config.TORCH_DTYPE),
        device_map=Config.DEVICE,
        load_in_4bit=True if Config.DEVICE == "cuda" else False,
        cache_dir="./models"
    )
    return processor, model
