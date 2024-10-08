"""
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name llava
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name llava-next
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name blip2 --max_length 1024 --max_tokens 700
CUDA_VISIBLE_DEVICES=3 python run_captioning.py --model_name kosmos2 --max_length 2048 --max_tokens 1500
"""

import os
import json
import argparse
import tiktoken
from tqdm import tqdm
from PIL import Image
import copy
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from model_loader import load_model, load_image, encode_image
from eval.utils import str2bool
from eval.utils import create_prompt, postprocess_output_new
from transformers.image_utils import load_image as tf_load_image
from qwen_vl_utils import process_vision_info

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)


def truncate(x, max_len):
    token_list = encoding.encode(x)[:max_len]
    return encoding.decode(token_list)

def token_len(x):
    return len(encoding.encode(x))


def format_prompt(abstract, content, with_abstract, with_content, max_tokens):
    
    assert not (with_abstract and with_content)
    input_content = ''
    if with_abstract:
        input_content += f'Article:\n{abstract}'
    if with_content:
        input_content += f'Article:\n{content}'

    if input_content:
        instruction = "Please write a detailed description of the whole figure and all sub-figures based on the article.\n"
        inst_len = token_len(instruction)
        input_content = truncate(input_content, max_tokens-inst_len)
        prompt = f"{instruction} {input_content}"
        if not prompt.endswith(('.', '!', '?')): # if truncate and is not a complete sentence.
            prompt = prompt + " ...\n" 
        if with_content:
            prompt += "\n" + instruction
    else:
        prompt = "Please write a detailed description of the whole figures and all sub-figures based on the article."

    return prompt



def _main(args):

    with_abstract, with_content = args.with_abstract, args.with_content
    assert not (with_abstract and with_content)
    tag = f"abs{with_abstract}_ctx{with_content}"

    # create output directory
    output_dir = os.path.join(args.base_output_dir, args.task, tag, f'k_{args.k}')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'{args.model_name}.json')
    print(f"Saving inference outputs for [{args.model_name}] to {output_filename}")

    # load data
    data = json.load(open(args.input_filename, 'r'))

    if os.path.exists(output_filename):
        with open(output_filename, "r") as file:
            output_list = json.load(file)
        print(f"Loading inference outputs for [{args.model_name}] to {output_filename}")
    else:
        output_list = copy.deepcopy(data)

    # load model
    if args.model_name == 'llava-next' or args.model_name.startswith('llava-ours'):
        tokenizer, model, image_processor, context_len = load_model(args.model_name)
    else:
        processor, model = load_model(args.model_name)

    # predict
    cnt = 0
    for item in tqdm(output_list, total=len(output_list), desc=tag):
        if args.model_name == 'llava-next' or args.model_name.startswith('llava-ours'):
            truncate_length = context_len - 300
            max_length = context_len
        elif args.model_name == "blip2":
            truncate_length = 300
            max_length = 512
        else:
            truncate_length = args.max_tokens
            max_length = args.max_length

        text = format_prompt(item['abstract'], item['content'], with_abstract, with_content, max_tokens=truncate_length)
        img_path = os.path.join(args.image_dir, item['image'])
        caption = item["caption"]
        # print(caption + '\n')
        # print(text + '\n')

        answers = [] if "prediction" not in item else item["prediction"]
        while len(answers) < args.k:
            try:
                if args.model_name == 'qwen':
                    query = processor.from_list_format([
                        {'image': img_path},
                        {'text': text},
                    ])
                    answer, history = model.chat(processor, query=query, history=None, temperature=args.temperature, top_p=args.top_p)
                    print(answer)

                elif args.model_name == 'llava-next' or args.model_name.startswith('llava-ours'):
                    conv_mode = "llava_v1"
                    conv = conv_templates[conv_mode].copy()
                
                    raw_image = Image.open(img_path).convert('RGB')
                    image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].half().cuda()
                    
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + text
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    out = model.generate(
                        inputs=input_ids, 
                        images=image_tensor,
                        do_sample=True, 
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=1024,
                    )
                    answer = postprocess_output_new(
                        model_name=args.model_name, 
                        output=tokenizer.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(answer+'\n')
                
                elif "internvl2" in args.model_name.lower():
                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)
                    question = '<image>\n' + text
                    answer = model.chat(processor, pixel_values, question, generation_config)
                    print(answer)

                elif "qwen2" in args.model_name.lower():
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": text},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    answer = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    print(answer)
                
                elif "llama3.2" in args.model_name.lower():
                    image = Image.open(img_path).convert('RGB')
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": text},
                        ]}
                    ]
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(image, input_text, return_tensors="pt").to(model.device)
                    output = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    answer = processor.decode(output[0])
                    print(answer)

                elif "idefics" in args.model_name.lower():
                    image = tf_load_image(img_path)
                    messages = [{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": text}
                    ]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[image], return_tensors="pt")
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    answer = generated_texts[0].split("Assistant:")[1].strip()
                    print(answer)
                
                elif "minicpm" in args.model_name.lower():
                    image = Image.open(img_path).convert('RGB')
                    question = text
                    msgs = [{'role': 'user', 'content': [image, question]}]
                    answer = model.chat(image=None, msgs=msgs, tokenizer=processor)
                    print(answer)

                else:
                    prompt = create_prompt(
                        model_name=args.model_name,
                        question=text,
                    )
                    raw_image = Image.open(img_path).convert('RGB')
                    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")

                    inputs['max_length'] = max_length
                    try:
                        out = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p)
                    except Exception as e:
                        print(f'Error: {e}')
                        continue
                    answer = postprocess_output_new(
                        model_name=args.model_name, 
                        output=processor.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(answer)
                    

            except Exception as e:
                print(e)
                answer = ""
               
            answers.append(answer)
            
        item['prediction'] = answers
    
        # save outputs
        with open(output_filename, 'w') as f:
            json.dump(output_list, f, indent=4)

        cnt += 1
        if cnt >= args.max_samples:
            break

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_filename', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/dev/image_caption_generation_data.json')
    argparser.add_argument('--base_output_dir', type=str, default='./eval/output/')
    argparser.add_argument('--image_dir', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/dev/images/')
    argparser.add_argument('--task', type=str, default='image_caption_generation')
    argparser.add_argument('--model_name', type=str, default='bert-base-uncased')
    argparser.add_argument('--max_tokens', type=int, default=1024)
    argparser.add_argument('--max_length', type=int, default=4096)
    argparser.add_argument('--max_samples', type=int, default=10e4)

    argparser.add_argument('--with_abstract', type=str2bool, default=True)
    argparser.add_argument('--with_content', type=str2bool, default=False)

    argparser.add_argument("--temperature", type=float, default=0.7)
    argparser.add_argument("--top_p", type=float, default=1.0)
    argparser.add_argument("--k", type=int, default=1, help="use self-consistency (majority voting) if > 1")

    args = argparser.parse_args()

    _main(args)
