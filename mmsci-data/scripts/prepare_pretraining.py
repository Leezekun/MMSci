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
from pylatexenc.latex2text import LatexNodes2Text
from io import BytesIO
import base64
import random
import copy

from subjects import subjects
from transformers import AutoTokenizer
from utils import *


def process_category(base_path, category, split_ids, subjects):
    
    processed_data = []
    # iterate each article directory
    for subject in subjects[category]:
        subject_path = os.path.join(base_path, category, subject)

        for entry in tqdm(os.listdir(subject_path)):
            full_path = os.path.join(subject_path, entry)
            if os.path.isdir(full_path):

                uid = os.path.basename(full_path)
                if uid not in split_ids[category][subject]:
                    continue

                try:
                    filename = os.path.join(full_path, f"{uid}_processed_data.json")
                    with open(filename, "r", encoding='utf-8') as file:
                        data = json.load(file)
                except:
                    continue
                    
                # start processing
                unique_id = data["unique_id"]
                article_link = f"https://www.nature.com/articles/{unique_id}"
                title = data["title"]
                abstract = data["abstract"]
                published_time = data["published_time"]

                text_list = []
                text_section = []
                processed_images = {}
                image_text_similarity = {}

                # process images
                for image in data["images"]:

                    image_filename = os.path.join(full_path, image["image_filename"])
                    description = image["description"]
                    caption = image["caption"]

                    # find figure index, starting from 1
                    figure_idx = find_integer_before_colon(caption)
                    if figure_idx == -1:
                        print(caption)
                        print("No image index found")
                        continue
                    
                    # record the image text similarity of each image
                    if figure_idx not in image_text_similarity:
                        image_text_similarity[figure_idx] = []

                    processed_image = {
                        "face_detections": None,
                        "index": figure_idx,
                        "image_name": f"{unique_id}_{figure_idx}.png",
                        "local_path": image_filename,
                        "raw_url": os.path.join(article_link, "articles", str(figure_idx)),
                        "caption": caption + " " + description,
                    }
                    processed_images[figure_idx] = processed_image

                # process title
                text_list.append(title)
                text_section.append("title")
                

                abs_sents = sent_tokenize(abstract)
                text_list.extend(abs_sents)
                text_section.extend(["abstract"]*len(abs_sents))

                # process main content
                for section in data["sections"]:
                    section_content = section["content"]
                    section_title = section["section"] 

                    sec_sents = sent_tokenize(section_content)
                    for sec_sent in sec_sents:
                        text_list.append(sec_sent)
                        text_section.append(section_title)

                if len(text_list) <= 1:
                    continue

                # check image_text_similarity
                for img_idx in image_text_similarity:
                    for sec_idx, sec_sent in enumerate(text_list):
                        # mentioned
                        if f"Figure {img_idx}" in sec_sent or f"Fig. {img_idx}" in sec_sent or \
                            f"Table {img_idx}" in sec_sent or f"Tab. {img_idx}" in sec_sent or \
                            f"Tab {img_idx}" in sec_sent or f"Fig {img_idx}" in sec_sent: 
                            image_text_similarity[img_idx].append(1.0)
                        else:
                            image_text_similarity[img_idx].append(0.)

                # insert images one by one
                img_indices = list(image_text_similarity.keys())
                img_indices = [int(i) for i in img_indices]
                img_indices.sort(reverse=False) # descending order
                matched_text_indices = [-1]*len(img_indices)
                for i, img_idx in enumerate(img_indices):
                    sim_vec = image_text_similarity[img_idx]
                    try: # found a matched one
                        matched_text_idx = sim_vec.index(1)
                    except ValueError:
                        matched_text_idx = -1
                    matched_text_indices[i] = matched_text_idx
                
                # process the txt idx to avoid not matching images
                for i, txt_idx in enumerate(matched_text_indices):
                    if txt_idx == -1:
                        if i == 0: # first one
                            matched_text_indices[i] = min(1, len(text_list))
                        elif i == len(matched_text_indices) - 1: # last one
                            matched_text_indices[i] = min(matched_text_indices[i-1], len(text_list)-1)
                        elif matched_text_indices[i+1] == -1:
                            next_valid_idx = -1
                            for next_idx in matched_text_indices[i:]:
                                if next_idx != -1:
                                    next_valid_idx = next_idx
                                    break
                            if next_valid_idx != -1:
                                matched_text_indices[i] = random.randint(min(matched_text_indices[i-1], next_valid_idx),
                                                                        max(matched_text_indices[i-1], next_valid_idx))
                            else:
                                matched_text_indices[i] = random.randint(matched_text_indices[i-1],len(text_list)-1)
                        else:
                            matched_text_indices[i] = random.randint(min(matched_text_indices[i-1],matched_text_indices[i+1]), 
                                                                    max(matched_text_indices[i-1],matched_text_indices[i+1]))

                image_text_matching = {}
                # start inserting image captions
                for i, img_idx in enumerate(img_indices):
                    caption = processed_images[img_idx]["caption"]
                    txt_idx = matched_text_indices[i]
                    text_list.insert(txt_idx, caption)
                    image_text_matching[img_idx] = txt_idx
                    # recauclate the indices of following figures
                    new_matched_text_indices = matched_text_indices[:i] + [idx+1 for idx in matched_text_indices[i:]]
                    matched_text_indices = new_matched_text_indices
                # recauculate the similarity matrix
                for i, img_idx in enumerate(img_indices):
                    for j, txt_idx in enumerate(new_matched_text_indices):
                        if i == j:
                            image_text_similarity[img_idx].insert(txt_idx, 1.0)
                        else:
                            image_text_similarity[img_idx].insert(txt_idx, 0.)

                # only keep the matched image and text
                matched_images = []
                similarity_matrix = []
                for img_index in image_text_matching:
                    if image_text_matching[img_index] and int(image_text_matching[img_index]) < len(text_list):
                        # process the image to image_base64
                        success = True
                        try:
                            img = Image.open(processed_images[img_index]["local_path"]).convert("RGB")
                            size_limit = 336  # reduce the resolution to save disk space
                            if min(img.size) > size_limit:
                                w, h = img.size
                                if h < w:
                                    new_h = size_limit
                                    new_w = int(size_limit * w / h)
                                else:
                                    new_w = size_limit
                                    new_h = int(size_limit * h / w)
                                img = img.resize((new_w, new_h))
                            
                            buffered = BytesIO()
                            img.save(buffered, format="JPEG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        except Exception as e:
                            print(e)
                            success = False
                        if success:
                            processed_image = processed_images[img_index]
                            processed_image["image_base64"] = img_b64_str
                            processed_image["matched_text_index"] = int(image_text_matching[img_index])
                            processed_image["matched_sim"] = 1.0 # string matching
                            matched_images.append(processed_image)
                            similarity = image_text_similarity[img_index]
                            assert len(similarity) == len(text_list)
                            similarity_matrix.append(similarity)
                
                if len(matched_images) > 0 and len(text_list) > 0:
                    dp = {
                        "image_info": matched_images,
                        "similarity_matrix": similarity_matrix,
                        "text_list": text_list,
                        "url": article_link,
                        "category": f"{category}-{subject}"
                    }
                    processed_data.append(dp)
            

    return processed_data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default="all") #
    args, unknown = parser.parse_known_args()

    base_path = "./rawdata"
    all_categories = list(subjects.keys())
    if args.category == "all":
        scraped_categories = all_categories
    else:
        assert args.category in all_categories
        scraped_categories = [args.category]

    base_path = "../rawdata"
    target_path = "../pretraindata"

    mmsci_all_train_data = []
    mmsci_all_dev_data = []
    mmsci_all_test_data = []
    
    for category in scraped_categories:
        mmsci_category_train_data = []
        mmsci_category_dev_data = []
        mmsci_category_test_data = []

        os.makedirs(os.path.join(target_path, category), exist_ok=True)

        processed_train_data_path = os.path.join(target_path, category, f"train_data.pkl")
        if not os.path.exists(processed_train_data_path):
            # load split ids
            train_split_id_path = os.path.join(base_path, "train_split_ids.json")
            with open(train_split_id_path, "r") as file:
                train_split_ids = json.load(file)

            mmsci_category_train_data = process_category(base_path, category, train_split_ids, subjects)
            # save category data
            with open(processed_train_data_path, "wb") as file:
                pickle.dump(mmsci_category_train_data, file)
        else:
            with open(processed_train_data_path, 'rb') as file:
                mmsci_category_train_data = pickle.load(file)
        mmsci_all_train_data.extend(mmsci_category_train_data)

        processed_dev_data_path = os.path.join(target_path, category, f"dev_data.pkl")
        if not os.path.exists(processed_dev_data_path):
            # load split ids
            dev_split_id_path = os.path.join(base_path, "dev_split_ids.json")
            with open(dev_split_id_path, "r") as file:
                dev_split_ids = json.load(file)

            mmsci_category_dev_data = process_category(base_path, category, dev_split_ids, subjects)
            # save category data
            with open(processed_dev_data_path, "wb") as file:
                pickle.dump(mmsci_category_dev_data, file)
        else:
            with open(processed_dev_data_path, 'rb') as file:
                mmsci_category_dev_data = pickle.load(file)
        mmsci_all_dev_data.extend(mmsci_category_dev_data)

        processed_test_data_path = os.path.join(target_path, category, f"test_data.pkl")
        if not os.path.exists(processed_test_data_path):
            # load split ids
            test_split_id_path = os.path.join(base_path, "test_split_ids.json")
            with open(test_split_id_path, "r") as file:
                test_split_ids = json.load(file)

            mmsci_category_test_data = process_category(base_path, category, test_split_ids, subjects)
            # save category data
            with open(processed_test_data_path, "wb") as file:
                pickle.dump(mmsci_category_test_data, file)
        else:
            with open(processed_test_data_path, 'rb') as file:
                mmsci_category_test_data = pickle.load(file)
        mmsci_all_test_data.extend(mmsci_category_test_data)

    # combine data from all the categories together and save
    if not os.path.exists(os.path.join(target_path, f"train_data.pkl")):
        with open(os.path.join(target_path, f"train_data.pkl"), "wb") as file:
            random.shuffle(mmsci_all_train_data)
            pickle.dump(mmsci_all_train_data, file)

    if not os.path.exists(os.path.join(target_path, f"dev_data.pkl")):
        with open(os.path.join(target_path, f"dev_data.pkl"), "wb") as file:
            random.shuffle(mmsci_all_dev_data)
            pickle.dump(mmsci_all_dev_data, file)

    if not os.path.exists(os.path.join(target_path, f"test_data.pkl")):
        with open(os.path.join(target_path, f"test_data.pkl"), "wb") as file:
            random.shuffle(mmsci_all_test_data)
            pickle.dump(mmsci_all_test_data, file)
        