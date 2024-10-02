import json
import requests
import os
import time
import PIL.Image
import base64
import random
from openai import OpenAI
from anthropic import Anthropic


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gemini_chat_completion(
    messages,
    n=1,
    max_tokens=1024,
    temperature=0.7,
    top_p=1.0,
    model="gemini-1.5-pro-001",
    api_key="GEMINI_API_KEY",
):
    assert n == 1

    gemini_prompt_setting = {
        "model": model,
        "apikey": api_key,
        "input": {},
        "generation_config": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "candidateCount": int(n),
        },
    }
    gemini_safety_settings = [
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    gemini_input_messages = []
    sys_prompt = ""
    for message in messages:
        role = message["role"]
        if role == "system":
            sys_prompt = message["content"]
        if role == "user":
            parts = []
            for c in message["content"]:
                if c["type"] == "text":
                    parts.append({"text": c["text"]})
                elif c["type"] == "image_url":
                    parts.append(
                        {
                            "inline_data": {
                                "data": encode_image(c["image_url"]["url"]),
                                "mime_type": "image/png",
                            }
                        }
                    )
                else:
                    raise ValueError

            gemini_input_messages.append({"role": "user", "parts": parts})
        if role == "assistant" or role == "model":
            gemini_input_messages.append(
                {"role": "model", "parts": {"text": message["content"]}}
            )

    gemini_system_instruction = {
        "parts": [
            {"text": f"{sys_prompt}"},
        ]
    }
    gemini_prompt_setting["input"]["contents"] = gemini_input_messages
    gemini_prompt_setting["input"]["system_instruction"] = gemini_system_instruction
    gemini_prompt_setting["input"]["safety_settings"] = gemini_safety_settings
    payload = json.dumps(gemini_prompt_setting)
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    ## Request gemini
    url = "http://34.149.115.40.nip.io/gemini"
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.text)

    outputs, retry = [], 0
    while not outputs and retry < 20:
        try:
            outputs = [
                candidate["content"]["parts"][0]["text"]
                for candidate in json.loads(response.text)["candidates"]
            ]
            # response_metainfo = json.loads(response.text)['usageMetadata']
            if outputs:
                return outputs
        except Exception as e:
            print(f"Errors: {e}")
            time.sleep(0.5)
            retry += 1
            outputs = []
    return outputs


def openai_chat_completion(
    messages,
    n=1,
    max_tokens=1024,
    temperature=0.7,
    top_p=1.0,
    model="gpt-4o-2024-05-13",
    api_key="OPENAI_API_KEY",
):
    client = OpenAI(api_key=api_key)

    for message in messages:
        if message["role"] == "user":
            if isinstance(message["content"], str):
                continue

            for c in message["content"]:
                if c["type"] == "image_url":
                    base64_image = encode_image(c["image_url"]["url"])
                    c["image_url"]["url"] = f"data:image/jpeg;base64,{base64_image}"

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
    )

    outputs = []
    for candidate in completion.choices:
        message = candidate.message.content
        outputs.append(message)
    return outputs


def claude_chat_completion(
    messages,
    n=1,
    max_tokens=1024,
    temperature=0.7,
    top_p=1.0,
    model="claude-3-5-sonnet-20240620",
    api_key="CLAUDE_API_KEY",
):
    assert n == 1
    client = Anthropic(api_key=api_key)

    message_list = []
    for message in messages:
        if message["role"] == "user":
            for idx, c in enumerate(message["content"]):
                if c["type"] == "image_url":
                    base64_image = encode_image(c["image_url"]["url"])
                    c = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    }
                    message["content"][idx] = c

        message_list.append(message)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=message_list,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = [response.content[0].text]
    return outputs


if __name__ == "__main__":
    # system_instruction = 'You will respond as a music historian, demonstrating comprehensive knowledge across diverse musical genres and providing relevant examples. Your tone will be upbeat and enthusiastic, spreading the joy of music. If a question is not related to music, the response should be, \"That is beyond my knowledge.\"'
    # input_text = 'If a person was born in the sixties, what was the most popular music genre being played when they were born? List five songs by bullet point.'
    # llm_gemini(input_text, system_instruction)

    QA_TRIGGER = '\n\nPlease first thoroughly analyze and think about this problem, and then come to your final answer. Conclude your response with: "The final answer is **[ANSWER]**."'

    input_text = """
Please write a detailed description of the whole figure and all sub-figures based on the article abstract.
Abstract: 
Cullin-RING ubiquitin ligases (CRLs) are critical in ubiquitinating Myc, while COP9 signalosome (CSN) controls neddylation of Cullin in CRL. The mechanistic link between Cullin neddylation and Myc ubiquitination/degradation is unclear. Here we show that Myc is a target of the CSN subunit 6 (CSN6)–Cullin signalling axis and that CSN6 is a positive regulator of Myc. CSN6 enhanced neddylation of Cullin-1 and facilitated autoubiquitination/degradation of Fbxw7, a component of CRL involved in Myc ubiquitination, thereby stabilizing Myc. Csn6 haplo-insufficiency decreased Cullin-1 neddylation but increased Fbxw7 stability to compromise Myc stability and activity in an Eμ-Myc mouse model, resulting in decelerated lymphomagenesis. We found that CSN6 overexpression, which leads to aberrant expression of Myc target genes, is frequent in human cancers. Together, these results define a mechanism for the regulation of Myc stability through the CSN–Cullin–Fbxw7 axis and provide insights into the correlation of CSN6 overexpression with Myc stabilization/activation during tumorigenesis.
"""
    input_image = "../mmsci-data/benchmark/dev/images/ncomms6384_figure_5.png"
    # input_image = "/home/ubuntu/MMSci/setting3.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": input_image}},
            ],
        }
    ]
    print(messages)
    output = openai_chat_completion(messages, model="gpt-4o")[0]
    # output = gemini_chat_completion(messages, model="gemini-1.5-pro-001")[0]
    # output = claude_chat_completion(messages, model="claude-3-5-sonnet-20240620")[0]
    print(output)

    # with open("../mmsci-data/benchmark/dev/image_caption_matching_data.json", "r") as file:
    #     data = json.load(file)

    # examples = []
    # for level, level_data in enumerate(data):
    #     level_example_subjects = {}
    #     random.shuffle(level_data)
    #     for dp in level_data:
    #         category = dp["category"]
    #         subject = dp["subject"]
    #         if subject in level_example_subjects:
    #             continue

    #         question = dp["question"]
    #         answer = dp["answer"]
    #         image_path = os.path.join("../mmsci-data/benchmark/dev/images", dp["image"])
    #         messages = messages = [
    #             {"role": "user", "content": [{"type": "text", "text": question+QA_TRIGGER}, {"type": "image_url", "image_url": {"url": image_path}}]}
    #         ]
    #         output = openai_chat_completion(messages,n=1)[0]

    #         # print(output)
    #         # _ = input("Press ENTER to continue......")

    #         for trigger in ["The final answer is **", "The final answer is "]:
    #             if trigger in output:
    #                 prediction = output.split(trigger)[1][0]
    #                 explanation = output.split(trigger)[0]
    #                 print(f"Prediction: {prediction}; Answer: {answer}")
    #                 print(f"Explanation: {explanation}")
    #                 break
    #         else:
    #             continue

    #         if prediction in ["A", "B", "C", "D"]:
    #             if prediction == answer:
    #                 dp["prediction"] = prediction
    #                 dp["explanation"] = explanation
    #                 level_example_subjects[subject] = dp
    #         else:
    #             print(output)

    #         if len(level_example_subjects) >= 10:
    #             break

    #     level_examples = []
    #     for subject, dp in level_example_subjects.items():
    #         level_examples.append(dp)

    #     examples.append(level_examples)

    # with open("./eval/prompts/cot_examples.json", "w") as file:
    #     json.dump(examples, file, indent=4)
