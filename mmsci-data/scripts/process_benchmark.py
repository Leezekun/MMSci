import os
import json
from tqdm import tqdm
import argparse
from PIL import Image
import re
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
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

def prcoess_benchmark_evaluation_data(raw_path, processed_path, split_ids, subjects, max_subject_size=10e9, tokenizer="meta-llama/Llama-2-7b-hf"):

    processed_generation_data_path = os.path.join(processed_path, "image_caption_generation_data.json")    
    processed_matching_data_path = os.path.join(processed_path, "image_caption_matching_data.json")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if os.path.exists(processed_generation_data_path) and os.path.exists(processed_matching_data_path):
        with open(processed_generation_data_path, "r") as file:
            processed_generation_data = json.load(file)
        with open(processed_matching_data_path, "r") as file:
            processed_matching_data = json.load(file)
        return processed_generation_data, processed_matching_data

    processed_generation_data = []
    processed_matching_data = [[], [], []]

    processed_image_path = os.path.join(processed_path, "images")
    if not os.path.exists(processed_image_path):
        os.makedirs(processed_image_path)

    num_tokens = []
    for category, subcategories in subjects.items():
        for subject in subcategories:
            subject_path = os.path.join(raw_path, category, subject)
            assert os.path.exists(subject_path)

            gen_data_size = 0
            match_data_size = [0,0,0]

            for entry in tqdm(os.listdir(subject_path)):
                full_path = os.path.join(subject_path, entry)
                if os.path.isdir(full_path):
                    uid = os.path.basename(full_path)
                    
                    if uid not in split_ids[category][subject]:
                        continue

                    # Process this article
                    try:
                        filename = os.path.join(full_path, f"{uid}_processed_data.json")
                        with open(filename, "r", encoding='utf-8') as file:
                            data = json.load(file)
                    except Exception as e:
                        print(e)
                        continue
                    
                    # Process this article
                    title = data["title"]
                    abstract = data["abstract"]
                    images = data["images"]

                    #################################################
                    #####   Task 1: Image caption generation   #####
                    ################################################
                    if gen_data_size <= max_subject_size:

                        text_list, text_section = [], []

                        # part 1: title
                        text_list.append(title)
                        text_section.append("title")

                        # part 2: abstract
                        text_list.append(abstract)
                        text_section.append("abstract")

                        # part 3: main content
                        for section in data["sections"]:
                            section_content = section["content"]
                            section_title = section["section"]
                            text_list.append(section_content)
                            text_section.append(section_title)
                        whole_content = "\n".join(text_list)
                        
                        valid_choices = []
                        for img_idx, image in enumerate(data["images"]):
                            caption_ = image["caption"]
                            caption_ = caption_.split(":")[::-1][0].strip()
                            description_ = image["description"]
                            caption = caption_ + " " + description_
                            if len(caption.split()) > MIN_CAPTION_WORDS:
                                valid_choices.append([img_idx, caption_, description_])
                        
                        if not valid_choices:
                            continue

                        # random select one for each article
                        selected_img = random.choice(valid_choices)
                        selected_img_idx = selected_img[0]
                        selected_img_caption = selected_img[1]
                        selected_img_description = selected_img[2]
                        image = data["images"][selected_img_idx]
                        
                        # check if the description in the whole text
                        whole_content = whole_content.replace(selected_img_caption, "")
                        whole_content = whole_content.replace(selected_img_description, "")
                        
                        img_caption = selected_img_caption + " " + selected_img_description
                        num_tokens.append(len(tokenizer(img_caption).input_ids))

                        # copy data
                        img_fname = image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        # save data
                        processed_generation_data.append(
                            {
                                "uid": uid,
                                "category": category,
                                "subject": subject,
                                "abstract": data["abstract"],
                                "content": whole_content,
                                "image": img_path,
                                "caption": img_caption
                            }
                        )
                        gen_data_size += 1

                    #################################################
                    #####   Task 2 (I): image caption matching #####
                    ################################################
                    if match_data_size[0] <= max_subject_size:

                        if len(data["images"]) < 4:
                            continue
                        
                        valid_choices = []
                        random.shuffle(data["images"])
                        for x in data["images"]:   
                            answer = find_caption(x["caption"])
                            # no answer extracted
                            if answer == x["caption"]:
                                continue
                            valid_choices.append(x)

                        if len(valid_choices) < 4:
                            continue
                        
                        for selected_image in valid_choices:
                            answer = find_caption(selected_image["caption"])
                            others = [item for item in valid_choices if item != selected_image]
                            others = [find_caption(x["caption"]) for x in others]
                            others = random.sample(others, 3)
                            choices_dict, answer_choice = convert_to_multichoice([answer]+others, answer=answer)
                            question = f"Which of the following captions best describes the whole figure?"
                            for choice, choice_content in choices_dict.items():
                                question += f"\n{choice}: {choice_content}"

                            # copy data
                            img_fname = selected_image["image_filename"]
                            img_path = f"{uid}_{img_fname}"
                            dst_img_path = os.path.join(processed_image_path, img_path)
                            src_image_path = os.path.join(full_path, img_fname)
                            if not os.path.exists(dst_img_path):
                                if os.path.exists(src_image_path):
                                    shutil.copyfile(src_image_path, dst_img_path)
                                else:
                                    continue

                            # save data
                            processed_matching_data[0].append(
                                {
                                    "uid": uid,
                                    "category": category,
                                    "subject": subject,
                                    "question": question,
                                    "answer": answer_choice,
                                    "image": f"{uid}_{img_fname}",
                                }
                            )
                            match_data_size[0] += 1
                            break

                    #################################################
                    #####   Task 2 (II): image caption matching #####
                    ################################################
                    if match_data_size[1] <= max_subject_size:
                        all_subcaptions = []
                        for image in data["images"]:
                            caption = image["caption"]
                            description = image["description"]

                            # find all subfigures and subcaptions
                            subcaptions = find_sub_caption(description)
                            all_subcaptions.extend(list(subcaptions.values()))
                        
                        for image in data["images"]:
                            caption = image["caption"]
                            description = image["description"]

                            # find all subfigures and subcaptions
                            subcaptions = find_sub_caption(description)

                            if len(subcaptions) <= 1:
                                continue

                            # randomly select a subfigure
                            subfigure = random.choice(list(subcaptions.keys()))
                            subcaption = subcaptions[subfigure]
                            
                            # add subcaption from other figures
                            choices = [subcaption]
                            random.shuffle(all_subcaptions)
                            for s in all_subcaptions:
                                if s not in choices:
                                    choices.append(s)
                                    if len(choices) == 4:
                                        break
                                    
                            if len(choices) != 4:
                                continue

                            # construct the question
                            question = f"which of the following options best describes the content in sub-figure ({subfigure})?"
                            choices_dict, answer_choice = convert_to_multichoice(choices, answer=subcaption)
                            for choice, choice_content in choices_dict.items():
                                question += f"\n{choice}: {choice_content}"

                            # copy data
                            img_fname = image["image_filename"]
                            img_path = f"{uid}_{img_fname}"
                            dst_img_path = os.path.join(processed_image_path, img_path)
                            src_image_path = os.path.join(full_path, img_fname)
                            if not os.path.exists(dst_img_path):
                                if os.path.exists(src_image_path):
                                    shutil.copyfile(src_image_path, dst_img_path)
                                else:
                                    continue

                            # save data
                            processed_matching_data[1].append(
                                {
                                    "uid": uid,
                                    "category": category,
                                    "subject": subject,
                                    "question": question,
                                    "answer": answer_choice,
                                    "image": f"{uid}_{img_fname}",
                                }
                            )
                            match_data_size[1] += 1
                            break

                    #################################################
                    ##### Task 2 (III): image caption matching #####
                    ################################################
                    if match_data_size[2] <= max_subject_size:
                        for image in data["images"]:
                            caption = image["caption"]
                            description = image["description"]

                            # find all subfigures and subcaptions
                            subcaptions = find_sub_caption(description)
                            if len(subcaptions) < 4:
                                continue
                            
                            # randomly select a subfigure
                            subfigure = random.choice(list(subcaptions.keys()))
                            subcaption = subcaptions[subfigure]

                            # construct the question
                            question = f"which of the following options best describes the content in sub-figure ({subfigure})?"
                            choices = list(subcaptions.values())
                            choices.remove(subcaption)
                            choices = random.sample(choices, 3)
                            choices.append(subcaption)
                            choices_dict, answer_choice = convert_to_multichoice(choices, answer=subcaption)
                            for choice, choice_content in choices_dict.items():
                                question += f"\n{choice}: {choice_content}"

                            # copy data
                            img_fname = image["image_filename"]
                            img_path = f"{uid}_{img_fname}"
                            dst_img_path = os.path.join(processed_image_path, img_path)
                            src_image_path = os.path.join(full_path, img_fname)
                            if not os.path.exists(dst_img_path):
                                if os.path.exists(src_image_path):
                                    shutil.copyfile(src_image_path, dst_img_path)
                                else:
                                    continue

                            # save data
                            processed_matching_data[2].append(
                                {
                                    "uid": uid,
                                    "category": category,
                                    "subject": subject,
                                    "question": question,
                                    "answer": answer_choice,
                                    "image": f"{uid}_{img_fname}",
                                }
                            )
                            match_data_size[2] += 1
                            break

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(num_tokens, bins=100, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Number of Tokens in Image Captions')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(processed_path, "num_tokens.png"))

    # save the data
    with open(processed_generation_data_path, "w") as file:
        json.dump(processed_generation_data, file, indent=4, ensure_ascii=False)    
    with open(processed_matching_data_path, "w") as file:
        json.dump(processed_matching_data, file, indent=4, ensure_ascii=False)   

    return processed_generation_data, processed_matching_data


def prcoess_benchmark_training_data(raw_path, processed_path, split_ids, subjects, tokenizer="meta-llama/Llama-2-7b-hf"):

    processed_generation_data_path = os.path.join(processed_path, "image_caption_generation_data.json")    
    processed_matching_data_path = os.path.join(processed_path, "image_caption_matching_data.json")
    processed_chat_data_path = os.path.join(processed_path, "image_caption_chat_data.json")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    ConvTemplate = Conversation()

    if os.path.exists(processed_generation_data_path) and os.path.exists(processed_matching_data_path) \
        and os.path.exists(processed_chat_data_path):
        with open(processed_generation_data_path, "r") as file:
            processed_generation_data = json.load(file)
        with open(processed_matching_data_path, "r") as file:
            processed_matching_data = json.load(file)
        with open(processed_chat_data_path, "r") as file:
            processed_chat_data = json.load(file)
        return processed_generation_data, processed_matching_data, processed_chat_data

    processed_generation_data = []
    processed_matching_data = [[], [], []] # Leval I, II, III
    processed_chat_data = []

    processed_image_path = os.path.join(processed_path, "images")
    if not os.path.exists(processed_image_path):
        os.makedirs(processed_image_path)

    for category, subcategories in subjects.items():
        for subject in subcategories:
            subject_path = os.path.join(raw_path, category, subject)
            assert os.path.exists(subject_path)

            for entry in tqdm(os.listdir(subject_path)):
                full_path = os.path.join(subject_path, entry)
                if os.path.isdir(full_path):
                    uid = os.path.basename(full_path)
                    
                    if uid not in split_ids[category][subject]:
                        continue

                    # Process this article
                    try:
                        filename = os.path.join(full_path, f"{uid}_processed_data.json")
                        with open(filename, "r", encoding='utf-8') as file:
                            data = json.load(file)
                    except Exception as e:
                        print(e)
                        continue
                    
                    # Process this article
                    title = data["title"]
                    abstract = data["abstract"]
                    images = data["images"]

                    #################################################
                    #####   Task 1: Image caption generation   #####
                    ################################################
                    text_list, text_section = [], []

                    # part 1: title
                    text_list.append(title)
                    text_section.append("title")

                    # part 2: abstract
                    text_list.append(abstract)
                    text_section.append("abstract")

                    # part 3: main content
                    for section in data["sections"]:
                        section_content = section["content"]
                        section_title = section["section"]
                        text_list.append(section_content)
                        text_section.append(section_title)
                    whole_content = "\n".join(text_list)
                    
                    image_generation_data = []
                    for img_idx, image in enumerate(data["images"]):
                        caption_ = image["caption"]
                        caption_ = caption_.split(":")[::-1][0].strip()
                        description_ = image["description"]
                        
                        # check if the description in the whole text
                        whole_content = whole_content.replace(caption_, "")
                        whole_content = whole_content.replace(description_, "")
                        
                        img_caption = caption_ + " " + description_

                        # copy data
                        img_fname = image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        image_generation_data.append({
                            "image": img_path, 
                            "caption": img_caption
                            })

                    # save data
                    processed_generation_data.append(
                        {
                            "uid": uid,
                            "category": category,
                            "subject": subject,
                            "abstract": data["abstract"],
                            "content": whole_content,
                            "images": image_generation_data
                        }
                    )

                    #################################################
                    #####   Task 2 (I): image caption matching #####
                    ################################################
                    if len(data["images"]) < 4:
                        continue

                    valid_choices = []
                    random.shuffle(data["images"])
                    for x in data["images"]:   
                        answer = find_caption(x["caption"])
                        # no answer extracted
                        if answer == x["caption"]:
                            continue
                        valid_choices.append(x)

                    if len(valid_choices) < 4:
                        continue
                    
                    # random select one
                    for selected_image in valid_choices:                   
                        answer = find_caption(selected_image["caption"])
                        others = [item for item in valid_choices if item != selected_image]
                        others = [find_caption(x["caption"]) for x in others]
                        others = random.sample(others, 3)
                        choices_dict, answer_choice = convert_to_multichoice([answer]+others, answer=answer)
                        question = f"Which of the following captions best describes the whole figure?"
                        for choice, choice_content in choices_dict.items():
                            question += f"\n{choice}: {choice_content}"

                        # copy data
                        img_fname = selected_image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        # save data
                        processed_matching_data[0].append(
                            {
                                "uid": uid,
                                "category": category,
                                "subject": subject,
                                "question": question,
                                "answer": answer_choice,
                                "image": f"{uid}_{img_fname}",
                            }
                        )

                    #################################################
                    #####   Task 2 (II): image caption matching #####
                    ################################################
                    all_subcaptions = []
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]

                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)
                        all_subcaptions.extend(list(subcaptions.values()))
                    
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]

                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)

                        if len(subcaptions) <= 1:
                            continue

                        # randomly select a subfigure
                        subfigure = random.choice(list(subcaptions.keys()))
                        subcaption = subcaptions[subfigure]
                        
                        # add subcaption from other figures
                        choices = [subcaption]
                        random.shuffle(all_subcaptions)
                        for s in all_subcaptions:
                            if s not in choices:
                                choices.append(s)
                                if len(choices) == 4:
                                    break
                                
                        if len(choices) != 4:
                            continue

                        # construct the question
                        question = f"which of the following options best describes the content in sub-figure ({subfigure})?"
                        choices_dict, answer_choice = convert_to_multichoice(choices, answer=subcaption)
                        for choice, choice_content in choices_dict.items():
                            question += f"\n{choice}: {choice_content}"

                        # copy data
                        img_fname = image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        # save data
                        processed_matching_data[1].append(
                            {
                                "uid": uid,
                                "category": category,
                                "subject": subject,
                                "question": question,
                                "answer": answer_choice,
                                "image": f"{uid}_{img_fname}",
                            }
                        )

                    #################################################
                    ##### Task 2 (III): image caption matching #####
                    ################################################
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]

                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)
                        if len(subcaptions) < 4:
                            continue
                        
                        # randomly select a subfigure
                        subfigure = random.choice(list(subcaptions.keys()))
                        subcaption = subcaptions[subfigure]

                        # construct the question
                        question = f"which of the following options best describes the content in sub-figure ({subfigure})?"
                        choices = list(subcaptions.values())
                        choices.remove(subcaption)
                        choices = random.sample(choices, 3)
                        choices.append(subcaption)
                        choices_dict, answer_choice = convert_to_multichoice(choices, answer=subcaption)
                        for choice, choice_content in choices_dict.items():
                            question += f"\n{choice}: {choice_content}"

                        # copy data
                        img_fname = image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        # save data
                        processed_matching_data[2].append(
                            {
                                "uid": uid,
                                "category": category,
                                "subject": subject,
                                "question": question,
                                "answer": answer_choice,
                                "image": f"{uid}_{img_fname}",
                            }
                        )

                    
                    #################################################
                    #####      Task 3: image caption CHAT      #####
                    ################################################
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]

                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)
                        if len(subcaptions) <= 1:
                            continue
                        
                        subfigures = list(subcaptions.keys())
                        num_subfigures = len(subfigures)
                        random.shuffle(subfigures)

                        conversation = []
                        conv_templates = ConvTemplate.get_conv_template(size=num_subfigures, concise_ratio=0.5)                    
                        for idx, subfigure in enumerate(subfigures):
                            conv_template = conv_templates[idx]
                            subcaption = subcaptions[subfigure]
                            conversation.append({"from": "human", "value": conv_template.format(subfigure)})
                            conversation.append({"from": "assistant", "value": subcaption})

                        # copy data
                        img_fname = image["image_filename"]
                        img_path = f"{uid}_{img_fname}"
                        dst_img_path = os.path.join(processed_image_path, img_path)
                        src_image_path = os.path.join(full_path, img_fname)
                        if not os.path.exists(dst_img_path):
                            if os.path.exists(src_image_path):
                                shutil.copyfile(src_image_path, dst_img_path)
                            else:
                                continue

                        # save data
                        processed_chat_data.append(
                            {
                                "uid": uid,
                                "category": category,
                                "subject": subject,
                                "image": f"{uid}_{img_fname}",
                                "caption": image["caption"],
                                "conversations": conversation,
                            }
                        )

    # save the data
    with open(processed_generation_data_path, "w") as file:
        json.dump(processed_generation_data, file, indent=4, ensure_ascii=False)    
    with open(processed_matching_data_path, "w") as file:
        json.dump(processed_matching_data, file, indent=4, ensure_ascii=False)   
    with open(processed_chat_data_path, "w") as file:
        json.dump(processed_chat_data, file, indent=4, ensure_ascii=False)   

    return processed_generation_data, processed_matching_data, processed_chat_data


def summarize_processed_data(data):
    # summarize the data
    num_samples = len(data)
    unique_articles = []

    summary = categorized_data(subjects, init_value={"articles": [], "samples": 0})
    for dp in data:
        category = dp["category"] 
        subject = dp["subject"]
        summary[category][subject]["samples"] += 1
        if dp["uid"] not in summary[category][subject]["articles"]:
            summary[category][subject]["articles"].append(dp["uid"])
        if dp["uid"] not in unique_articles:
            unique_articles.append(dp["uid"])

    num_articles = len(unique_articles)
    for category, subcategories in summary.items():
        for subject in subcategories:
            summary[category][subject]["articles"] = len(summary[category][subject]["articles"])
    summary["Total"] = {
        "samples": num_samples,
        "article": num_articles,
        "sample_per_article": num_samples/num_articles
    }
    print(json.dumps(summary, indent=4))


if __name__ == '__main__':
    
    base_path = "../rawdata"
    target_path = "../benchmark"

     # # prepare test set
    processed_test_datapath = os.path.join(target_path, "test")
    if not os.path.exists(processed_test_datapath):
        os.makedirs(processed_test_datapath)
    # load split ids
    test_split_id_path = os.path.join(base_path, "test_split_ids.json")
    with open(test_split_id_path, "r") as file:
        test_split_ids = json.load(file)
    # process data
    processed_caption_data, processed_matching_data = prcoess_benchmark_evaluation_data(base_path, processed_test_datapath, test_split_ids, subjects, MAX_SUBJECT_SAMPLE)
    # summarize data
    print("Summarizing test image caption generation data .........")
    summarize_processed_data(processed_caption_data)
    for i in range(3):
        print(F"Summarizing dev image caption matching {'I'*i} data .........")
        summarize_processed_data(processed_matching_data[i])

    # # prepare dev set
    processed_dev_datapath = os.path.join(target_path, "dev")
    if not os.path.exists(processed_dev_datapath):
        os.makedirs(processed_dev_datapath)
    # load split ids
    dev_split_id_path = os.path.join(base_path, "dev_split_ids.json")
    with open(dev_split_id_path, "r") as file:
        dev_split_ids = json.load(file)
    # process data
    processed_caption_data, processed_matching_data = prcoess_benchmark_evaluation_data(base_path, processed_dev_datapath, dev_split_ids, subjects)
    # summarize data
    print("Summarizing dev image caption generation data .........")
    summarize_processed_data(processed_caption_data)
    for i in range(3):
        print(F"Summarizing dev image caption matching {'I'*i} data .........")
        summarize_processed_data(processed_matching_data[i])

    # prepare train set
    processed_train_datapath = os.path.join(target_path, "train")
    if not os.path.exists(processed_train_datapath):
        os.makedirs(processed_train_datapath)
    # load split ids
    train_split_id_path = os.path.join(base_path, "train_split_ids.json")
    with open(train_split_id_path, "r") as file:
        train_split_ids = json.load(file)
    # process data
    processed_caption_data, processed_matching_data, processed_chat_data = prcoess_benchmark_training_data(base_path, processed_train_datapath, train_split_ids, subjects)
    print("Summarizing train image caption generation data .........")
    summarize_processed_data(processed_caption_data)

    for i in range(3):
        print(F"Summarizing dev image caption matching {i} data .........")
        summarize_processed_data(processed_matching_data[i])

    print("Summarizing train image caption chat data .........")
    summarize_processed_data(processed_chat_data)
