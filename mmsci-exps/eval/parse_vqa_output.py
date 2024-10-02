import json
import re
import os
from collections import Counter
from tqdm import tqdm
import requests


def openai_parse_answer(question, answer):
    api_key='xxx'

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    model = 'gpt-3.5-turbo-0125'

    payload = {
    "model": model,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f'Question: {question}\Response: {answer}\n\nGiven this response to the question, what is the final answer? Answer only with A, or B, or C, or D, or Unknown.',
            }
        ]
        }
    ],
    "max_tokens": 10,
    "n": 1,
    "temperature": 0.1,
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        answer = response["choices"][0]["message"]["content"].strip()
        for a in ["A", "B", "C", "D", "Unknown"]:
            if a in answer:
                return a
        else:
            return answer
    except:
        return 'wrong answer'

k = 5
base_dir = '/mnt/raid0/zekun/MMSci/mmsci-exps/eval/output/image_caption_matching'

for cot in [True, False]:
    for setting in [1, 2, 3]:
        rst_dir = os.path.join(base_dir, "w_cot" if cot else "wo_cot", f"setting-{setting}", f"k_{k}")

        for file in ["gpt-4-turbo.json"]: # gpt-4o.json, kosmos2.json, llava-next-mistral.json, llava.json, llava-next.json, qwen.json, llava-next-mmsci.json

            rst_list = json.load(open(os.path.join(rst_dir, file)))

            ans_file = os.path.join(rst_dir, file.replace(".json", "-w-ans.json"))
            if not os.path.exists(ans_file):
                ans_list = rst_list
            else:
                ans_list = json.load(open(ans_file))
                if len(ans_list) != len(rst_list):
                    ans_list = rst_list

            num_total = len(ans_list)
            num_correct = [0, 0, 0] # 1, 3, 5
            valid = [0, 0, 0] # 1, 3, 5
            for item in tqdm(ans_list):
                # counter = Counter
                gt = item["answer"]
                question = item['question']
                preds = [] if 'prediction' not in item else item['prediction'] # only select the top_k pred for evaluation
                extracted_preds = []
                for pred in preds:
                    if "extracted_answer" not in pred:
                        extracted_answer = openai_parse_answer(question, pred["answer"])
                        print(f"GT: {gt}, PRED: {extracted_answer}")
                        pred["extracted_answer"] = extracted_answer
                        extracted_preds.append(extracted_answer)
                with open(ans_file, "w") as file:
                    json.dump(ans_list, file, indent=4)