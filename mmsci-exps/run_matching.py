import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import copy

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from eval.utils import create_prompt, postprocess_output
from model_loader import load_model, load_image, encode_image
from transformers.image_utils import load_image as tf_load_image
from qwen_vl_utils import process_vision_info

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_filename', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_matching_data.json')
argparser.add_argument('--base_output_dir', type=str, default='./eval/output/')
argparser.add_argument('--image_dir', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/images/')
argparser.add_argument('--task', type=str, default='image_caption_matching')
argparser.add_argument('--model_name', type=str, default='bert-base-uncased')
argparser.add_argument('--setting', type=int, default=3, choices=[1,2,3,4])

argparser.add_argument("--temperature", type=float, default=0.7)
argparser.add_argument("--top_p", type=float, default=1.0)
argparser.add_argument("--max_tokens", type=int, default=512)

# cot/sc-cot
argparser.add_argument("--cot", action="store_true", default=False, help="whether to use chain-of-thought prompting")
argparser.add_argument("--k", type=int, default=1, help="use self-consistency (majority voting) if > 1")

args = argparser.parse_args()

COT_PROMPT = "\n\nPlease first thoroughly analyze and think about this problem, and then come to your final answer."
COT_TRIGGER = "\nBefore we dive into the answer, "
ANS_TRIGGER = "\nTherefore, the final answer is:"
QA_TRIGGER = '\nChoose one answer from available options. Return only your answer.'
ANS_MAX_TOKENS = 128

if __name__ == '__main__':
    # create output directory
    w_cot = 'w_' if args.cot else 'wo_'
    output_dir = os.path.join(args.base_output_dir, args.task, f'{w_cot}cot', f'setting-{args.setting}', f'k_{args.k}')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'{args.model_name}.json')
    print(f"Saving inference outputs for [{args.model_name}] to {output_filename}")

    # load data
    data = json.load(open(args.input_filename, 'r'))[args.setting-1]

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
    for item in tqdm(output_list, total=len(output_list), desc='predicting'):
        img_path = os.path.join(args.image_dir, item['image'])

        answers = [] if "prediction" not in item else item["prediction"]

        while len(answers) < args.k:
            
            explanation = ""

            if args.model_name == 'qwen':
                if not args.cot:
                    query = processor.from_list_format([
                        {'image': img_path},
                        {'text': item['question']+QA_TRIGGER},
                    ])
                    answer, history = model.chat(processor, query=query, history=None, temperature=args.temperature)
                    print(answer)
                else:
                    # round 1: explanation
                    query = processor.from_list_format([
                        {'image': img_path},
                        {'text': item['question']+COT_PROMPT},
                    ])
                    explanation, history = model.chat(processor, query=query, history=None, temperature=args.temperature)
                    print(explanation)

                    # round 2: answer
                    answer, history = model.chat(processor, ANS_TRIGGER, history=history, temperature=args.temperature)
                    print(answer)

            elif args.model_name == 'llava-next' or args.model_name.startswith('llava-ours'):
                conv_mode = "llava_v1"
            
                raw_image = Image.open(img_path).convert('RGB')
                image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].half().cuda()
                
                if not args.cot:
                    conv = conv_templates[conv_mode].copy()
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + item['question'] + QA_TRIGGER
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
                        max_new_tokens=ANS_MAX_TOKENS,
                    )
                    answer = postprocess_output(
                        model_name=args.model_name, 
                        output=tokenizer.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(answer)

                else:
                    # round 1: explanation
                    conv = conv_templates[conv_mode].copy()
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + item['question'] + COT_PROMPT
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], COT_TRIGGER)
                    prompt = conv.get_prompt(generate=True)
                
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    out = model.generate(
                        inputs=input_ids, 
                        images=image_tensor,
                        do_sample=True, 
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_tokens,
                    )
                    explanation = postprocess_output(
                        model_name=args.model_name, 
                        output=tokenizer.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(prompt+"\n")
                    print(explanation)

                    # round 2: answer
                    conv = conv_templates[conv_mode].copy()
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + item['question'] + COT_PROMPT
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], explanation + ANS_TRIGGER)
                    prompt = conv.get_prompt(generate=True)
                
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    out = model.generate(
                        inputs=input_ids, 
                        images=image_tensor,
                        do_sample=True, 
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=ANS_MAX_TOKENS,
                    )
                    answer = postprocess_output(
                        model_name=args.model_name, 
                        output=tokenizer.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(prompt+"\n")
                    print(answer)
            
            elif "qwen2" in args.model_name.lower():
                if not args.cot:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": item['question'] + QA_TRIGGER},
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
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": item['question']},
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
                    rationale = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    messages.extend([
                        {"role": "assistant", "content": [{"type": "text", "text": rationale}]}, 
                        {"role": "user", "content": [{"type": "text", "text": ANS_TRIGGER}]}])
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
                if not args.cot:
                    image = Image.open(img_path).convert('RGB')
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": item['question']+QA_TRIGGER},
                        ]}
                    ]
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(image, input_text, return_tensors="pt").to(model.device)
                    output = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    answer = processor.decode(output[0])
                    print(answer)
                else:
                    image = Image.open(img_path).convert('RGB')
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": item['question']+COT_PROMPT},
                        ]}
                    ]
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(image, input_text, return_tensors="pt").to("cuda")
                    output = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    explanation = processor.decode(output[0])
                    print(explanation)

                    messages.extend([
                        {"role": "assistant", "content": explanation},
                        {"role": "user", "content": ANS_TRIGGER},
                    ])
                    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(image, input_text, return_tensors="pt").to("cuda")
                    output = model.generate(**inputs, max_new_tokens=ANS_MAX_TOKENS)
                    answer = processor.decode(output[0])
                    print(answer)

            elif "internvl2" in args.model_name.lower():
                if not args.cot:
                    question = '<image>\n' + item['question'] + QA_TRIGGER
                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)
                    answer = model.chat(processor, pixel_values, question, generation_config)
                    print(answer)
                else:
                    raise ValueError

            elif "idefics" in args.model_name.lower():
                image = tf_load_image(img_path)
                if not args.cot:
                    text = item['question'] + QA_TRIGGER 
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
                else:
                    text = item['question'] + COT_PROMPT 
                    messages = [{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": text}
                    ]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[image], return_tensors="pt")
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    answer = generated_texts[0].split("Assistant:")[-1].strip()
                    messages.extend([{"role": "assistant", "content": answer}, {"role": "user", "content": ANS_TRIGGER}])
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[image], return_tensors="pt")
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    answer = generated_texts[0].split("Assistant:")[-1].strip()
                    print(answer)

            elif "minicpm" in args.model_name.lower():
                image = Image.open(img_path).convert('RGB')
                question = item['question'] + QA_TRIGGER
                msgs = [{'role': 'user', 'content': [image, question]}]
                answer = model.chat(image=None, msgs=msgs, tokenizer=processor)
                print(answer)

            else:
                raw_image = Image.open(img_path).convert('RGB')

                if not args.cot:
                    prompt = create_prompt(
                        model_name=args.model_name,
                        question=item['question']+QA_TRIGGER,
                    )
                    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
                    inputs['max_new_tokens'] = ANS_MAX_TOKENS

                    out = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p)
                    answer = postprocess_output(
                        model_name=args.model_name, 
                        output=processor.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(answer)
                else:
                    # round 1: explanation
                    prompt = create_prompt(
                        model_name=args.model_name,
                        question=item['question']+COT_PROMPT,
                        answer=COT_TRIGGER
                    )
                    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
                    inputs['max_new_tokens'] = args.max_tokens

                    out = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p)
                    explanation = postprocess_output(
                        model_name=args.model_name, 
                        output=processor.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(prompt+"\n")
                    print(explanation)

                    # round 2: answer
                    prompt = create_prompt(
                        model_name=args.model_name,
                        question=item['question']+COT_PROMPT,
                        answer=explanation+ANS_TRIGGER
                    )
                    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
                    inputs['max_new_tokens'] = ANS_MAX_TOKENS

                    out = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p)
                    answer = postprocess_output(
                        model_name=args.model_name, 
                        output=processor.decode(out[0], skip_special_tokens=True).strip(),
                    )
                    print(prompt+"\n")
                    print(answer)
                    
            answers.append({"answer": answer, "explanation": explanation})
        
        # save the outputs
        item['prediction'] = answers
        
        # save outputs
        with open(output_filename, 'w') as f:
            json.dump(output_list, f, indent=4)
