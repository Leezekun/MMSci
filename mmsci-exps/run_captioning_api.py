"""
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name llava
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name llava-next
CUDA_VISIBLE_DEVICES=1 python run_captioning.py --model_name blip2 --max_length 1024 --max_tokens 700
CUDA_VISIBLE_DEVICES=3 python run_captioning.py --model_name kosmos2 --max_length 2048 --max_tokens 1500
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import copy
import torch
from eval.utils import str2bool
from llm_utils import *
import tiktoken

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

    if "gpt" in args.model_name:
        client = openai_chat_completion
    elif "claude" in args.model_name:
        client = claude_chat_completion
    elif "gemini" in args.model_name:
        client = gemini_chat_completion
    else:
        raise ValueError

    # predict
    cnt = 0
    for item in tqdm(output_list, total=len(output_list), desc=tag):
        
        truncate_length = args.max_tokens
        max_length = args.max_length

        text = format_prompt(item['abstract'], item['content'], with_abstract, with_content, max_tokens=truncate_length)
        img_path = os.path.join(args.image_dir, item['image'])
        caption = item["caption"]
        # print(caption + '\n')
        # print(text + '\n')

        answers = [] if "prediction" not in item else item["prediction"]

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": img_path}}
                    ]
                }
            ]

        while len(answers) < args.k:
            try:
                answer = client(messages,n=args.k, max_tokens=args.max_tokens)[0]
                print(answer)
            except Exception as e:
                print(f"Failed to generate response: {e}")
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
    argparser.add_argument('--input_filename', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_generation_data.json')
    argparser.add_argument('--base_output_dir', type=str, default='./eval/output/')
    argparser.add_argument('--image_dir', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/images/')
    argparser.add_argument('--task', type=str, default='image_caption_generation')
    argparser.add_argument('--model_name', type=str, default='bert-base-uncased')
    argparser.add_argument('--max_tokens', type=int, default=3000)
    argparser.add_argument('--max_length', type=int, default=4096)
    argparser.add_argument('--max_samples', type=int, default=10e4)

    argparser.add_argument('--with_abstract', type=str2bool, default=True)
    argparser.add_argument('--with_content', type=str2bool, default=False)

    argparser.add_argument("--temperature", type=float, default=0.7)
    argparser.add_argument("--top_p", type=float, default=1.0)
    argparser.add_argument("--k", type=int, default=1, help="use self-consistency (majority voting) if > 1")

    args = argparser.parse_args()

    _main(args)
