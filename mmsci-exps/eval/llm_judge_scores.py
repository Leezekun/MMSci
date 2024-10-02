'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
import argparse
from tqdm import tqdm
import numpy as np
import os
import json
import pprint
import warnings
from collections import defaultdict
from openai import OpenAI
import time
import random
import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluator',
        type=str,
        choices=['openai', 'gemini'],
        help='Model for evaluation.')

    parser.add_argument(
        '--model',
        type=str,
        help='Model generation for evaluation.')

    parser.add_argument(
        '--input_json',
        type=str,
        help='Candidates json mapping from image_id --> candidate.')
    
    parser.add_argument(
        '--base_generation_output_dir',
        default='./output/image_caption_generation',
        type=str,
    )

    parser.add_argument(
        '--k',
        default=1, 
        type=int
    )

    parser.add_argument(
        '--overwrite',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


def llm_gemini_judge(prediction, reference, dimension, prompt, model="gemini-1.5-pro-exp-0801", api_key="GEMINI_API_KEY"):
    
    if isinstance(reference, list):
        reference = reference[0]
    assert isinstance(prediction, str)

    if dimension == "fluency":
        sys_prompt = prompt.format(Second=prediction) # no reference
        usr_prompt = f"Caption:\n{prediction}\n"
    else:
        sys_prompt = prompt.format(Oracle=reference, Second=prediction)
        usr_prompt = f"Oracle Caption:\n{reference}\n\nSecond Caption:\n{prediction}\n\n"
    usr_prompt += f"What is the {dimension} score (1-5)? Return the score ONLY!"

    gemini_prompt_setting = {
        "model": model,
        "apikey": api_key,
        "input": { },
        "generation_config": {
            "maxOutputTokens": 8192,
            "temperature": 1,
            "topP": 0.95
        }
    }
    gemini_safety_settings = [
            {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
            },
            {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
            }
    ]
    gemini_input_messages = {
                "role": "user",
                "parts": {
                    "text": f"{usr_prompt}"
                }
    }
    gemini_system_instruction = {
        "parts": [
            {
                "text": f"{sys_prompt}"
            },
        ]
    }
    gemini_prompt_setting['input']['contents'] = gemini_input_messages
    gemini_prompt_setting['input']['system_instruction'] = gemini_system_instruction
    gemini_prompt_setting['input']['safety_settings']=gemini_safety_settings
    payload = json.dumps(gemini_prompt_setting)
    headers = {'Content-Type': 'application/json','User-Agent': 'Mozilla/5.0'}
    ## Request gemini
    url = "http://34.149.115.40.nip.io/gemini"
    response = requests.request("POST", url, headers=headers, data=payload)

    answer, retry = 0, 0
    while answer == 0 and retry < 20:
        try:
            response = json.loads(response.text)['candidates'][0]['content']['parts'][0]['text']
            # response_metainfo = json.loads(response.text)['usageMetadata']
            try:
                if float(response) <= 5 and float(response) >= 1:
                    answer = float(response)
                    print(dimension + ": " + str(answer))
                    break
            except:
                continue
            break

        except Exception as e:       
            print(f"Errors: {e}")
            time.sleep(0.5)
            retry += 1
    return answer


def llm_openai_judge(prediction, reference, dimension, prompt, model="gpt-4o-2024-08-06", api_key="OPENAI_API_KEY"):
    client = OpenAI(api_key=api_key)
    
    if isinstance(reference, list):
        reference = reference[0]
    assert isinstance(prediction, str)

    if dimension == "fluency":
        sys_prompt = prompt.format(Second=prediction) # no reference
        usr_prompt = f"Caption:\n{prediction}\n"
    else:
        sys_prompt = prompt.format(Target=reference, Second=prediction)
        usr_prompt = f"Oracle Caption:\n{reference}\n\nSecond Caption:\n{prediction}\n\n"
    usr_prompt += f"What is the {dimension} score (1-5)? Return the score ONLY!"
    
    answer, retry = 0, 0
    while answer == 0 and retry < 5:
        try:
            all_responses = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt}
                    ],
                temperature=1,
                max_tokens=16,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=10 # multiple outputs
            )
            
            for response in all_responses.choices:
                response = response.message.content.strip()
                try:
                    answer = int(response)
                    break
                except:
                    for s in ["1", "2", "3", "4", "5"]:
                        if s in response:
                            answer = int(s)
                            break
        except Exception as e:
            print(e)
            time.sleep(0.5)
            retry += 1
    return answer


def get_all_metrics(references, candidates, evaluator):
    # Initialize the metrics

    if evaluator == "openai":
        llm_judge = llm_openai_judge
    elif evaluator == "gemini":
        llm_judge = llm_gemini_judge
    else:
        raise ValueError

    # coh_prompt = open("./prompt/coh_detailed.txt", "r").read()
    # flu_prompt = open("./prompts/flu_detailed.txt", "r").read()
    rel_prompt = open("./prompts/rel_detailed.txt", "r").read()
    con_prompt = open("./prompts/con_detailed.txt", "r").read()
    cap_prompt = open("./prompts/cap_detailed.txt", "r").read()

    # flu_scores = []
    # con_scores = [] # consistency scores
    rel_scores = [] # relevance scores
    # cap_scores = []

    # Calculate scores for each sample
    for ref, cand in tqdm(zip(references, candidates)):

        if not cand.strip():
            print(cand)
            continue

        # fluency score
        # flu_score = llm_judge(prediction=cand, reference=ref, dimension="fluency", prompt=flu_prompt)
        # flu_scores.append(flu_score)

        # consistency score
        # con_score = llm_judge(prediction=cand, reference=ref, dimension="consistency", prompt=con_prompt)
        # con_scores.append(con_score)

        # relevance score
        rel_score = llm_judge(prediction=cand, reference=ref, dimension="relevance", prompt=rel_prompt)
        print(rel_score)
        rel_scores.append(rel_score)

        # caption score
        # cap_score = llm_judge(prediction=cand, reference=ref, dimension="caption", prompt=cap_prompt)
        # cap_scores.append(cap_score)

    # Calculate average scores
    # avg_flu_score = np.mean(flu_scores)
    # avg_con_score = np.mean(con_scores)
    avg_rel_score = np.mean(rel_scores)
    # avg_cap_score = np.mean(cap_scores)

    # Create a dictionary to store the average scores
    metrics = {
        # "fluency": avg_flu_score,
        # "consistency": avg_con_score,
        "relevance": avg_rel_score,
        # "caption": avg_cap_score
    }
    
    return metrics


def main(args):

    print(f'Evaluating {args.input_json}')

    args.base_score_dir = f"./eval_scores/image_caption_generation/llm-{args.evaluator}-judge"
    score_dir = os.path.join(args.base_score_dir, args.tag)
    os.makedirs(score_dir, exist_ok=True)
    score_file = os.path.join(score_dir, args.input_json.split('/')[-1])
    print(score_file)
    if os.path.exists(score_file) and not args.overwrite:
        print(f'Already evaluated. Will skip...')
        return

    args.answer_path = f"../../mmsci-data/benchmark/test/image_caption_generation_data_w_answer.json"
    with open(args.answer_path, "r") as f:
        answer_data = json.load(f)

    answers = {}
    for dp in answer_data:
        answers[dp["abstract"]] = dp["caption"]

    with open(args.input_json) as f:
        data = json.load(f)
    random.shuffle(data)

    cnt = 0
    # reformat data
    old_candidates = {}
    old_references = {}
    for item in data:
        if 'prediction' not in item:
            continue

        caption = item['caption']
        if not caption:
            caption = answers[item['abstract']]

        image_id = item['image'].split('.')[0]
        old_candidates[image_id] = item['prediction']
        old_references[image_id] = [caption]

        cnt += 1
        if cnt >= 100:
            break

    candidates = []
    references = []
    image_ids = old_references.keys()
    for cid in image_ids:
        if cid in old_candidates:
            candidates.append(old_candidates[cid][args.eval_pred_idx])
            references.append(old_references[cid])

    if isinstance(references[0], str):
        references = [[r] for r in references]

    avg_scores = {}
    metrics = get_all_metrics(references, candidates, evaluator=args.evaluator)
    for k, v in metrics.items():
        avg_scores[k] = v

        if args.save_per_instance:
            with open(args.save_per_instance, 'w') as f:
                f.write(json.dumps(scores))

    return score_file, avg_scores


if __name__ == '__main__':

    args = parse_args()
    base_output_dir = '/home/ubuntu/MMSci/mmsci-exps/eval/output/image_caption_generation'

    for w_abs in [True]:
        for w_ctx in [False]:
            if w_abs and w_ctx:
                continue
            if "gpt" not in args.model and w_ctx:
                continue
                
            tag = f'abs{w_abs}_ctx{w_ctx}'
            args.tag = tag
            k = 3
            file = f"{args.model}.json"
            args.input_json = os.path.join(base_output_dir, tag, f"k_{k}", file)

            all_scores = defaultdict(list)
            for i in range(k):
                args.eval_pred_idx = i
                score_file, scores = main(args)
                print(scores)
                for metric, score in scores.items():
                    all_scores[metric].append(score)
            with open(score_file, 'w') as fout:
                json.dump(all_scores, fout, indent=4)
