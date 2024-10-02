import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import copy

from llm_utils import (
    gemini_chat_completion,
    claude_chat_completion,
    openai_chat_completion,
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--input_filename",
    type=str,
    default="/home/ubuntu/MMSci/mmsci-data/benchmark/test/image_caption_matching_data.json",
)
argparser.add_argument(
    "--example_filename",
    type=str,
    default="/home/ubuntu/MMSci/mmsci-exps/eval/prompts/cot_examples.json",
)
argparser.add_argument("--base_output_dir", type=str, default="./eval/output/")
argparser.add_argument(
    "--image_dir",
    type=str,
    default="/home/ubuntu/MMSci/mmsci-data/benchmark/test/images/",
)
argparser.add_argument("--task", type=str, default="image_caption_matching")
argparser.add_argument("--model_name", type=str, default="gemini-1.5-pro-001")
argparser.add_argument("--setting", type=int, default=1, choices=[1, 2, 3, 4])

argparser.add_argument("--temperature", type=float, default=0.7)
argparser.add_argument("--top_p", type=float, default=1.0)
argparser.add_argument("--max_tokens", type=int, default=1024)

argparser.add_argument("--max_samples", type=int, default=10e4)

# cot/sc-cot
argparser.add_argument(
    "--cot",
    action="store_true",
    default=False,
    help="whether to use chain-of-thought prompting",
)
argparser.add_argument(
    "--n_shot", type=int, default=0, help="the number of examples for cot"
)
argparser.add_argument(
    "--k", type=int, default=1, help="use self-consistency (majority voting) if > 1"
)

args = argparser.parse_args()

COT_PROMPT = '\n\nPlease first thoroughly analyze and think about this problem, and then come to your final answer. Conclude your response with: "The final answer is **[ANSWER]**."'
COT_TRIGGER = "Before we dive into the answer, "
QA_TRIGGER = 'Analyze the problem before concluding your response in this format: "The final answer is **[ANSWER]**."'

ANS_MAX_TOKENS = 16

if __name__ == "__main__":
    # create output directory
    w_cot = "w_" if args.cot else "wo_"
    output_dir = os.path.join(
        args.base_output_dir,
        args.task,
        f"{w_cot}cot-{args.n_shot}shot",
        f"setting-{args.setting}",
        f"k_{args.k}",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{args.model_name}.json")
    print(f"Saving inference outputs for [{args.model_name}] to {output_filename}")

    if "gpt" in args.model_name:
        client = openai_chat_completion
    elif "claude" in args.model_name:
        client = claude_chat_completion
    elif "gemini" in args.model_name:
        client = gemini_chat_completion
    else:
        raise ValueError
    print(client)

    # load data for this level
    data = json.load(open(args.input_filename, "r"))[args.setting - 1]
    try:
        examples = json.load(open(args.example_filename, "r"))[args.setting - 1]
    except:
        examples = []
        
    if os.path.exists(output_filename):
        with open(output_filename, "r") as file:
            output_list = json.load(file)
        print(f"Loading inference outputs for [{args.model_name}] to {output_filename}")
    else:
        output_list = copy.deepcopy(data)

    # predict
    finished = 0
    for item in tqdm(output_list, total=len(output_list), desc="predicting"):
        img_path = os.path.join(args.image_dir, item["image"])

        answers = [] if "prediction" not in item else item["prediction"]
        # clean the invalid answer
        # for answer in answers:
        #     if answer["answer"] not in ["A", "B", "C", "D"]:
        #         answers.remove(answer)
        #         print("Remove:", json.dumps(answer, indent=4))

        retry = 0
        while not answers and retry < 5:
            if not args.cot:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item["question"] + QA_TRIGGER},
                            {"type": "image_url", "image_url": {"url": img_path}},
                        ],
                    }
                ]
            else:
                messages = []
                for example in examples:
                    if len(messages) // 2 <= args.n_shot:
                        break

                    messages.extend(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": example["question"] + COT_PROMPT,
                                },
                                {"type": "image_url", "image_url": {"url": img_path}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": example["explanation"]
                            + f"The final answer is **{example['prediction']}**.",
                        },
                    )

                # Current query
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item["question"] + COT_PROMPT},
                            {"type": "image_url", "image_url": {"url": img_path}},
                        ],
                    }
                )

            try:
                outputs = client(messages, n=args.k, model=args.model_name)
            except Exception as e:
                print(f"Failed to generate response: {e}")
                outputs = []

            print(outputs)

            # start inference
            for output in outputs:
                for trigger in ["The final answer is **", "The final answer is "]:
                    if trigger in output:
                        prediction = output.split(trigger)[1][0]
                        explanation = output.split(trigger)[0]
                        answers.append(
                            {"answer": prediction, "explanation": explanation}
                        )
                        break

            if not answers:
                print(outputs)

            retry += 1

        # save the outputs
        item["prediction"] = answers

        # save outputs
        with open(output_filename, "w") as f:
            json.dump(output_list, f, indent=4)

        finished += 1

        if finished >= args.max_samples:
            break
