import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import copy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from eval.utils import create_prompt, postprocess_output
from llava_builder import load_model

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_filename', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_matching_data.json')
argparser.add_argument('--base_output_dir', type=str, default='./eval/output/')
argparser.add_argument('--image_dir', type=str, default='/home/ubuntu/MMSci/mmsci-data/benchmark/test/images/')
argparser.add_argument('--task', type=str, default='image_caption_matching')
argparser.add_argument('--model_name', type=str, default='bert-base-uncased')
argparser.add_argument('--setting', type=int, default=3, choices=[1,2,3])

argparser.add_argument("--temperature", type=float, default=0.7)
argparser.add_argument("--top_p", type=float, default=1.0)
argparser.add_argument("--max_tokens", type=int, default=1024)

# cot/sc-cot
argparser.add_argument("--cot", action="store_true", default=False, help="whether to use chain-of-thought prompting")
argparser.add_argument("--k", type=int, default=1, help="use self-consistency (majority voting) if > 1")

args = argparser.parse_args()

COT_PROMPT = "\n\nPlease first thoroughly analyze and think about this problem, and then come to your final answer."
COT_TRIGGER = "Before we dive into the answer, "
ANS_TRIGGER = " Therefore, the final answer is (choose one from A/B/C/D):"
QA_TRIGGER = ' Choose one answer from A/B/C/D.'
ANS_MAX_TOKENS = 16

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
    if args.model_name == 'llava-next' or args.model_name.startswith('llava-ours') or 'mmsci' in args.model_name:
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

            elif args.model_name == 'llava-next' or args.model_name.startswith('llava-ours') or 'mmsci' in args.model_name:
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
                    inputs['max_new_tokens'] = argparse.max_tokens

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
