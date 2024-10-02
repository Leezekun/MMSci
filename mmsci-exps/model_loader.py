#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
Part of code derived from (https://github.com/haotian-liu/LLaVA)
@article{liu2024visual,
  title={Visual instruction tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={Advances in neural information processing systems},
  volume={36},
  year={2024}
}
"""

import os
import warnings
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
)
from transformers import (
    AutoProcessor,
    AutoModelForVisualQuestionAnswering,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    LlavaForConditionalGeneration,
    Kosmos2ForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation import GenerationConfig
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration
# from transformers import MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import base64
from llava.model import *
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model(model_name):
    models = {
        "blip2": "Salesforce/blip2-opt-2.7b",
        "llava": "llava-hf/llava-1.5-7b-hf",
        "llava-next": "liuhaotian/llava-v1.6-vicuna-7b",
        "llava-next-mistral": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llama3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "kosmos2": "microsoft/kosmos-2-patch14-224",
        "qwen": "Qwen/Qwen-VL-Chat",
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "internvl2-1b": "OpenGVLab/InternVL2-1B",
        "internvl2-2b": "OpenGVLab/InternVL2-2B",
        "internvl2-4b": "OpenGVLab/InternVL2-4B",
        "internvl2-8b": "OpenGVLab/InternVL2-8B",
        "internvl2-26b": "OpenGVLab/InternVL2-26B",
        "idefics2-8b": "HuggingFaceM4/idefics2-8b",
        "idefics3-8b": "HuggingFaceM4/Idefics3-8B-Llama3",
        "minicpm": "openbmb/MiniCPM-V-2_6",
        "florence2": "microsoft/Florence-2-large",
        "yi-vl-6b": "01-ai/Yi-VL-6B",
        # 'yi-vl-34b': '01-ai/Yi-VL-34B',
        "qwen2-vl-2b-mmsci-mixed": "/mnt/raid0/zekun/LLaMA-Factory/models/qwen2_vl_2b_lora_sft-mmsci-mixed",
        "qwen2-vl-2b-mmsci-mixed-v2": "/mnt/raid0/zekun/LLaMA-Factory/models/qwen2_vl_2b_lora_sft-mmsci-mixed-v2",
        "llava-next-mmsci": "/mnt/raid0/zekun/MMSci/mmsci-exps/checkpoints/llava-v1.6-vicuna-7b-mmsci",
        "llava-next-arxivqa": "/mnt/raid0/zekun/MMSci/mmsci-exps/checkpoints/llava-v1.6-vicuna-7b-arxivqa",
    }

    if model_name == "blip2":
        processor = Blip2Processor.from_pretrained(models[model_name])
        model = Blip2ForConditionalGeneration.from_pretrained(models[model_name])
    elif model_name == "llava":
        processor = AutoProcessor.from_pretrained(models[model_name])
        model = LlavaForConditionalGeneration.from_pretrained(models[model_name])
    elif model_name == "kosmos2":
        processor = AutoProcessor.from_pretrained(models[model_name])
        model = Kosmos2ForConditionalGeneration.from_pretrained(models[model_name])
    elif model_name == "llava-next-mistral":
        processor = LlavaNextProcessor.from_pretrained(models[model_name])
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    elif model_name == "llava-next":
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=models[model_name],
            model_name=model_name,
            model_base=None,
        )
        model = model.to("cuda")
        return tokenizer, model, image_processor, context_len
    # elif "llama3.2" in model_name:
    #     model = MllamaForConditionalGeneration.from_pretrained(
    #         models[model_name],
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #     )
    #     processor = AutoProcessor.from_pretrained(models[model_name])
    elif model_name == "qwen":
        processor = AutoTokenizer.from_pretrained(
            models[model_name], trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            models[model_name], device_map="cuda", trust_remote_code=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            models[model_name], trust_remote_code=True
        )
    elif "qwen2" in model_name.lower():
        print(models[model_name])
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            models[model_name],
            torch_dtype="auto",
            # device_map="auto"
        ).cuda()
        if "72b" in model_name:
            processor = AutoProcessor.from_pretrained(models["qwen2-vl-72b"])
        elif "7b" in model_name:
            processor = AutoProcessor.from_pretrained(models["qwen2-vl-7b"])
        elif "2b" in model_name:
            processor = AutoProcessor.from_pretrained(models["qwen2-vl-2b"])
    elif "internvl2" in model_name.lower():
        model = (
            AutoModel.from_pretrained(
                models[model_name],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            )
            .eval()
            .cuda()
        )
        processor = AutoTokenizer.from_pretrained(
            models[model_name], trust_remote_code=True, use_fast=False
        )
    elif "idefics" in model_name.lower():
        model = AutoModelForVision2Seq.from_pretrained(models[model_name]).cuda()
        processor = AutoProcessor.from_pretrained(models[model_name])
    elif "minicpm" in model_name.lower():
        model = AutoModel.from_pretrained(
            models[model_name],
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )  # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        processor = AutoTokenizer.from_pretrained(
            models[model_name], trust_remote_code=True
        )
    else:
        processor = AutoProcessor.from_pretrained(models[model_name])
        model = AutoModelForVisualQuestionAnswering.from_pretrained(models[model_name])

    model = model.to("cuda")
    return processor, model


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    if "llava" in model_name.lower():
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig

            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(
                        token_num, tokem_dim, device=model.device, dtype=model.dtype
                    )
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id, filename=filename, subfolder=subfolder
                    )
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(
                    model_path, "non_lora_trainables.bin"
                )
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_projector_weights = torch.load(
                os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
            )
            mm_projector_weights = {
                k: v.to(torch.float16) for k, v in mm_projector_weights.items()
            }
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            elif "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
            else:
                print("load from here...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, **kwargs
                )

    image_processor = None

    if "llava" in model_name.lower():
        print("will load image processor...")
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
