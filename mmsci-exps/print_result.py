import json
import os
from tqdm import tqdm
import random
from llm_utils import openai_chat_completion

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def rule_based_answer(answer, options):
    for a in options:
        if a == answer:
            return a
        if f"*Answer*: {a}" in answer or f"*Answer*: {a.lower()}" in answer:
            return a
        if f"**Answer**: {a}" in answer or f"**Answer**: {a.lower()}" in answer:
            return a
        if f" {a} " in answer:
            return a
        if f"sub-figure {a.lower()}" in answer:
            return a
        if f"subfigure {a.lower()}" in answer:
            return a
        if f"sub-figure ({a.lower()})" in answer:
            return a
        if f"subfigure ({a.lower()})" in answer:
            return a
        if f"{a}:" in answer:
            return a
        if f" {a}**" in answer:
            return a
        if f"**{a}**" in answer or f"**{a.lower()}**" in answer:
            return a
        if f" {a}." in answer or f" {a.lower()}." in answer:
            return a
    return answer


def extract_answer(question, answer, options, prioritize_llm=False):
    answer = answer.strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[-1].strip()

    if not prioritize_llm:
        answer = rule_based_answer(answer, options)
        if answer in options:
            return answer

    user_input = (
        "The analysis is as follow:\n" + answer
        if not question
        else f"The question is: {question}.\nThe anaylysis is: {answer}"
    )

    messages = [
        {
            "role": "system",
            "content": "The user will provide an analysis of a multiple-choice question with a few options. Based on the analysis, infer the correct answer.",
        },
        {
            "role": "user",
            "content": user_input
            + "\nWhat is the correct answer mentioned in this analysis? Return only the final choice (A, B, C, D, ...)",
        },
    ]
    # print(messages)
    answer = openai_chat_completion(messages=messages, model="gpt-4o-mini")[0]
    answer = answer.strip()
    print(answer)
    # _ = input("Press ENTER to continue....")

    answer = rule_based_answer(answer, options)
    if answer in options:
        return answer

    return answer


# Load the answer data
answer_path = "../mmsci-data/benchmark/test/image_caption_matching_data_w_answer.json"
with open(answer_path, "r") as fp:
    answer_data = json.load(fp)

# Prepare the answers dictionary
answers = {}
for level_data in answer_data:
    for dp in level_data:
        answers[dp["question"]] = dp["answer"]

level = 1
# Set the base path for human evaluation results
# base_path = f"./eval/output/image_caption_matching/w_cot/setting-{level}/k_5/gpt-4o-w-ans-v2.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot-0shot/setting-{level}/k_1/gpt-4-turbo.json"
# base_path = f"./eval/output/image_caption_matching/w_cot/setting-{level}/k_5/gpt-4o.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot-0shot/setting-{level}/k_1/gpt-4-turbo.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot-0shot/setting-{level}/k_1/gpt-4o.json"
# base_path = f"./eval/output/image_caption_matching/w_cot-0shot/setting-{level}/k_1/gemini-1.5-pro-001.json"
# base_path = f"./eval/output/image_caption_matching/w_cot-0shot/setting-{level}/k_1/claude-3-opus-20240229.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot-0shot/setting-{level}/k_1/claude-3-5-sonnet-20240620.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/minicpm.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/idefics2-8b.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/internvl2-8b.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/idefics3-8b.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/internvl2-1b.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/llama3.2-11b.json"
base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/qwen2-vl-2b.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/qwen2-vl-2b-mmsci-mixed-v2.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/qwen.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_5/llava-next-mmsci-mixed-1080k.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_5/llava-next-mistral.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/llava-next.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_1/kosmos2.json"
# base_path = f"./eval/output/image_caption_matching/wo_cot/setting-{level}/k_5/llava.json"
# base_path = f"./eval/output/image_caption_matching/w_cot-0shot/setting-{level}/k_1/gemini-1.5-flash-002.json"

# Iterate through each file in the base path
with open(base_path, "r") as fp:
    data = json.load(fp)

subjects = {}

# category = "Physical sciences"
# category = "Earth and environmental sciences"
# category = "Biological sciences"
# category = "Health sciences"
# category = "Scientific community and society"
category = None

random_guess = False
use_existing_answer = True

# Process each data point
correct = 0
total = 0
wrong_cases = []
subjects = {}
for idx, dp in tqdm(enumerate(data)):
    if category is not None and dp["category"] != category:
        continue

    subject = dp["subject"]
    question = dp["question"]
    gt_answer = answers[question]

    if random_guess:
        options = []
        for label in labels:
            if f"\n{label}: " in question:
                options.append(label)
        final_answer = random.choice(options)

    else:
        if "prediction" not in dp:
            continue
        if len(dp["prediction"]) < 1:
            continue

        if "explanation" not in dp["prediction"][0]:
            dp["prediction"][0]["explanation"] = dp["prediction"][0]["answer"]
        elif dp["prediction"][0]["explanation"] == "":
            dp["prediction"][0]["explanation"] = dp["prediction"][0]["answer"]

        # answer = dp["prediction"][0]["answer"]
        answer = "" if not use_existing_answer else dp["prediction"][0]["answer"]
        explanation = dp["prediction"][0]["explanation"]
        if answer not in labels:
            final_answer = extract_answer(
                question, explanation, labels, prioritize_llm=True
            )
        else:
            final_answer = answer

        if final_answer not in labels:
            print(f"Invalid answer: {answer}")
            continue
        else:
            dp["prediction"][0]["answer"] = final_answer

    # break down to the subject level
    if subject not in subjects:
        subjects[subject] = [0, 0]

    # Check if the answer matches the correct answer
    # print(answer)
    if final_answer == gt_answer:
        print(final_answer, gt_answer, correct)
        correct += 1
        subjects[subject][0] += 1
    else:
        wrong_cases.append(
            {
                "idx": idx,
                "category": dp["category"],
                "subject": subject,
                "image": dp["image"],
                "question": question,
                "gt_answer": gt_answer,
                "answer": final_answer,
                "explanation": explanation,
            }
        )

    # Count the occurrences for each subject
    subjects[subject][1] += 1
    total += 1

# save the extracted answers
if not use_existing_answer and not random_guess:
    with open(base_path, "w") as fp:
        json.dump(data, fp, indent=4)

print(base_path)
print(f"Level: {level}")
# Display the count for each subject
print(f"Total {len(subjects)} subjects, evaluated {total} data points")
for subject, count in subjects.items():
    accuracy = count[0] / count[1] if count[1] > 0 else 0
    print(f"{subject}:", f"{accuracy:.2%} ({count[0]}/{count[1]})")

# Calculate and print accuracy
accuracy = correct / total if total > 0 else 0
print(f"Total accuracy: {accuracy:.2%} ({correct}/{total})")