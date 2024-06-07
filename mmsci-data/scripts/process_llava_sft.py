import os
import json
from tqdm import tqdm
import argparse
from PIL import Image
import re

import pickle
import random
import shutil
import copy
import random
import glob
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from subjects import subjects
from utils import *
from conversation import Conversation


def process_chat_data(base_path, processed_path, target_path):

    with open(processed_path, "r") as file:
        processed_chat_data = json.load(file)

    with open(os.path.join(base_path, "valid_images.json")) as file:
        valid_images = json.load(file)

    llava_data = []
    for idx, data in tqdm(enumerate(processed_chat_data)):

        image_path = data["image"]
        if image_path not in valid_images:
            print(image_path)
            continue

        # add <image>\n at the beginning of the conversation
        conversations = data["conversations"]
        for conv in conversations:
            assert "from" in conv and "value" in conv
            if conv["from"] == "assistant":
                conv["from"] = "gpt"
        conversations[0]["value"] = "<image>\n" + conversations[0]["value"]
        data["conversations"] = conversations
        llava_data.append(data)
        
    random.shuffle(llava_data)
    with open(target_path, "w") as file:
        json.dump(llava_data, file, indent=4, ensure_ascii=False)   

    return llava_data


def process_matching_data(base_path, processed_path, target_path, settings=[1,2,3]):

    with open(processed_path, "r") as file:
        processed_matching_data = json.load(file)
    
    with open(os.path.join(base_path, "valid_images.json")) as file:
        valid_images = json.load(file)
    
    llava_data = []
    for setting in settings:
        setting_processed_matching_data = processed_matching_data[setting-1]
        # use at most 2 sample per article for setting=1
        max_sample_per_article = 2 if setting == 1 else 10e9
        unique_uids = {}

        for idx, data in tqdm(enumerate(setting_processed_matching_data)):

            image_path = data["image"]
            if image_path not in valid_images:
                print(image_path)
                continue
            
            uid = data["uid"]
            if uid in unique_uids:
                if unique_uids[uid] >= max_sample_per_article:
                    continue

            # single-turn conversaiton
            question = data["question"]
            answer = data["answer"]
            choices = question.split("\nA:")[1]

            # extract the content
            if answer == "A":
                answer_choice = choices.split("\nB:")[0].strip()
            elif answer == "B":
                answer_choice = choices.split("\nC:")[0].split("\nB:")[1].strip()
            elif answer == "C":
                answer_choice = choices.split("\nD:")[0].split("\nC:")[1].strip()
            elif answer == "D":
                answer_choice = choices.split("\nD:")[1].strip()

            human = "<image>\n" + question
            response = f"{answer}: {answer_choice}"

            conversations = [
                {"from": "human", "value": human},
                {"from": "gpt", "value": response}
            ]

            llava_data.append(
                {
                    "uid": data["uid"],
                    "category": data["category"],
                    "subject": data["subject"],
                    "image": data["image"],
                    "conversations": conversations
                }
            )
            
            if uid in unique_uids:
                unique_uids[uid] += 1
            else:
                unique_uids[uid] = 1

        print(setting, len(llava_data))

    random.shuffle(llava_data)
    with open(target_path, "w") as file:
        json.dump(llava_data, file, indent=4, ensure_ascii=False)   

    return llava_data


def process_caption_data(base_path, processed_path, target_path):

    ConvTemplate = Conversation()

    with open(os.path.join(base_path, "valid_images.json")) as file:
        valid_images = json.load(file)

    with open(processed_path, "r") as file:
        processed_caption_data = json.load(file)
    
    llava_data = []
    for idx, data in tqdm(enumerate(processed_caption_data)):

        # get info, change conversation
        abstract = data["abstract"]
        content = data["content"]

        for image in data["images"]:

            image_path = image["image"]
            if image_path not in valid_images:
                print(image_path)
                continue
            
            # single-turn conversaiton
            caption = image["caption"]
            human = ConvTemplate.get_detailed_caption()
            human = "<image>\n" + human.format("Article", abstract)

            conversations = [
                {"from": "human", "value": human},
                {"from": "gpt", "value": caption}
            ]

            llava_data.append(
                {
                    "uid": data["uid"],
                    "category": data["category"],
                    "subject": data["subject"],
                    "image": image["image"],
                    "conversations": conversations
                }
            )

    random.shuffle(llava_data)
    with open(target_path, "w") as file:
        json.dump(llava_data, file, indent=4, ensure_ascii=False)   

    return llava_data


def summarize_processed_data(data):
    # summarize the data
    num_samples = len(data)
    unique_articles = []

    summary = categorized_data(subjects, init_value=0)
    for dp in data:
        category = dp["category"] 
        subject = dp["subject"]
        summary[category][subject] += 1
        
    summary["Total"] = num_samples
    print(json.dumps(summary, indent=4))


if __name__ == '__main__':
    
    base_path = "../rawdata"
    target_path = "../benchmark"
    processed_train_data_path = os.path.join(target_path, "train")

    processed_chat_data_path = os.path.join(processed_train_data_path, "image_caption_chat_data.json")
    target_chat_data_path = os.path.join(processed_train_data_path, "llava_image_caption_chat_data.json")
    if os.path.exists(target_chat_data_path):
        with open(target_chat_data_path, "r") as file:
            processed_chat_data = json.load(file)
    else:
        processed_chat_data = process_chat_data(processed_train_data_path, processed_chat_data_path, target_chat_data_path)
    print(len(processed_chat_data))
    print("Summarizing train image caption chat data .........")
    summarize_processed_data(processed_chat_data)

    processed_caption_data_path = os.path.join(processed_train_data_path, "image_caption_generation_data.json")
    target_caption_data_path = os.path.join(processed_train_data_path, "llava_image_caption_generation_data.json")
    if os.path.exists(target_caption_data_path):
        with open(target_caption_data_path, "r") as file:
            processed_caption_data = json.load(file)
    else:
        processed_caption_data = process_caption_data(processed_train_data_path, processed_caption_data_path, target_caption_data_path)
    print(len(processed_caption_data))
    print("Summarizing train image caption chat data .........")
    summarize_processed_data(processed_caption_data)

    processed_matching_data_path = os.path.join(processed_train_data_path, "image_caption_matching_data.json")
    target_matching_data_path = os.path.join(processed_train_data_path, "llava_image_caption_matching_data.json")
    if os.path.exists(target_matching_data_path):
        with open(target_matching_data_path, "r") as file:
            processed_matching_data = json.load(file)
    else:
        processed_matching_data = process_matching_data(processed_train_data_path, processed_matching_data_path, target_matching_data_path)
    print(len(processed_matching_data))
    print("Summarizing train image caption chat data .........")
    summarize_processed_data(processed_matching_data)

    target_mixed_data_path = os.path.join(processed_train_data_path, "llava_image_caption_mixed_data.json")
    if os.path.exists(target_mixed_data_path):
        with open(target_mixed_data_path, "r") as file:
            mixed_data = json.load(file)
    else:
        mixed_data = processed_chat_data + processed_caption_data + processed_matching_data
        random.shuffle(mixed_data)
        with open(target_mixed_data_path, "w") as file:
            json.dump(mixed_data, file, indent=4, ensure_ascii=False)   
    print(len(mixed_data))
    summarize_processed_data(mixed_data)
