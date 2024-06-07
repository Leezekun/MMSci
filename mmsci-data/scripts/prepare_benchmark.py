import os
import json
from tqdm import tqdm
import random
import random
import glob

from subjects import subjects
from utils import *


def precheck_data_distribution(base_path, subjects):
    """
    Prepare train / dev / test splits
    Priority: test > dev > train
    """
    # ids
    summaries = categorized_data(subjects)

    # check
    for category in subjects:
        for subject in subjects[category]:
            subject_path = os.path.join(base_path, category, subject)

            # shuffle the dirs 
            all_dirs = os.listdir(subject_path)
            random.shuffle(all_dirs)

            summary = [0, 0, 0, 0, 0, 0] # valid1, valid2, valid3, valid4, valid5, total

            for entry in tqdm(all_dirs):
                full_path = os.path.join(subject_path, entry)
                if os.path.isdir(full_path):
                    uid = os.path.basename(full_path)

                    # start processing the data
                    try:
                        filename = os.path.join(full_path, f"{uid}_processed_data.json")
                        with open(filename, "r", encoding='utf-8') as file:
                            data = json.load(file)
                    except Exception as e:
                        print(e)
                        continue

                    # Check 1: has images
                    if data["images"]:
                        summary[0] += 1
                    
                    # Check 2: TASK 1: Image caption based on abstract or the whole content
                    for image in data["images"]:
                        caption_ = image["caption"]
                        caption_ = caption_.split(":")[::-1][0].strip()
                        description_ = image["description"]
                        caption = caption_ + " " + description_
                        if len(caption.split()) > MIN_CAPTION_WORDS:
                            # print(caption)
                            summary[1] += 1
                            break 
                            
                    # Check 3: TASK 2 (I): Image caption based on abstract or the whole content
                    if len(data["images"]) >= 4:
                        summary[2] += 1
                    
                    # Check 4: TASK 2 (II): Image caption based on abstract or the whole content
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]
                        
                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)
                        if len(subcaptions) >= 2 and len(data["images"]) >= 4:
                            summary[3] += 1

                    # Check 5: TASK 2 (II): Image caption based on abstract or the whole content
                    for image in data["images"]:
                        caption = image["caption"]
                        description = image["description"]
                        
                        # find all subfigures and subcaptions
                        subcaptions = find_sub_caption(description)
                        if len(subcaptions) >= 4:
                            summary[4] += 1
                    
                    summary[5] += 1

            summaries[category][subject] = summary
            print(summary)

    print_dict_with_indent(summaries)


def prepare_benchmark_data_split(base_path, subjects, avoid_ids, min_subject_size=MIN_SUBJECT_SAMPLE, ratio=0.01, select=True):
    
    # ids
    split_ids = categorized_data(subjects, init_value=[])
    task1_samples = categorized_data(subjects, init_value=0)
    task2_samples = categorized_data(subjects, init_value=0)
    all_articles = categorized_data(subjects, init_value=0)

    # check
    for category in subjects:
        for subject in subjects[category]:
            subject_path = os.path.join(base_path, category, subject)
            
            # shuffle the dirs 
            all_dirs = os.listdir(subject_path)
            total_size = len(all_dirs)
            all_articles[category][subject] = total_size

            for task_idx in range(2):
                random.shuffle(all_dirs)
                for entry in tqdm(all_dirs):
                    full_path = os.path.join(subject_path, entry)
                    if os.path.isdir(full_path):
                        uid = os.path.basename(full_path)

                        if uid in avoid_ids[category][subject]:
                            continue

                        # Start processing the data
                        try:
                            filename = os.path.join(full_path, f"{uid}_processed_data.json")
                            with open(filename, "r", encoding='utf-8') as file:
                                data = json.load(file)
                        except Exception as e:
                            print(e)
                            continue

                        if not select:
                            if uid not in split_ids[category][subject]:
                                split_ids[category][subject].append(uid)
                        else: # select test/dev data
                            # TASK 2 (II): Image caption based on abstract or the whole content
                            if task_idx == 0:
                                for image in data["images"]:
                                    caption = image["caption"]
                                    description = image["description"]
                                    
                                    # find all subfigures and subcaptions
                                    subcaptions = find_sub_caption(description)
                                    if len(subcaptions) >= 4:
                                        if uid not in split_ids[category][subject]:
                                            split_ids[category][subject].append(uid)
                                        task2_samples[category][subject] += 1
                            
                            # TASK 1: Image caption based on abstract or the whole content
                            if task_idx == 0 and uid in split_ids[category][subject]:
                                for image in data["images"]:
                                    caption_ = image["caption"]
                                    caption_ = find_caption(caption_)
                                    description_ = image["description"]
                                    caption = caption_ + " " + description_
                                    if len(caption.split()) > MIN_CAPTION_WORDS:
                                        task1_samples[category][subject] += 1
                                        break

                            if task_idx == 1 and uid not in split_ids[category][subject]:
                                for image in data["images"]:
                                    caption_ = image["caption"]
                                    caption_ = find_caption(caption_)
                                    description_ = image["description"]
                                    caption = caption_ + " " + description_
                                    if len(caption.split()) > MIN_CAPTION_WORDS:
                                        split_ids[category][subject].append(uid)
                                        task1_samples[category][subject] += 1
                                        break

                            # Check
                            if len(split_ids[category][subject]) >= int(ratio*total_size):
                                if task_idx == 0 and task2_samples[category][subject] >= min_subject_size:
                                    break
                                elif task_idx == 1 and task1_samples[category][subject] >= min_subject_size:
                                    break
            
    # Final summary
    summary = {}
    total_selected_articles = 0
    total_articles = 0
    total_task1_samples = 0
    total_task2_samples = 0
    for category in subjects:
        summary[category] = {}
        for subject in subjects[category]:
            summary[category][subject] = {
                "selected_articles": len(split_ids[category][subject]),
                "total_article": all_articles[category][subject],
                "task1_samples": task1_samples[category][subject],
                "task2_samples": task2_samples[category][subject]
            }
            total_selected_articles += len(split_ids[category][subject])
            total_articles += all_articles[category][subject]
            total_task1_samples += task1_samples[category][subject]
            total_task2_samples += task2_samples[category][subject]

    summary["Total"] = {
        "selected_articles": total_selected_articles,
        "total_article": total_articles,
        "task1_samples": total_task1_samples,
        "task2_samples": total_task2_samples
    }
    print_dict_with_indent(summary)

    return split_ids


if __name__ == '__main__':
    
    base_path = "../rawdata"
    target_path = "../benchmark"
    
    # precheck_data_distribution(base_path, subjects)

    train_split_id_path = os.path.join(base_path, "train_split_ids.json")
    dev_split_id_path = os.path.join(base_path, "dev_split_ids.json")
    test_split_id_path = os.path.join(base_path, "test_split_ids.json")

    if not (os.path.exists(train_split_id_path) and \
        os.path.exists(dev_split_id_path) and \
        os.path.exists(test_split_id_path)):

        exist_ids = categorized_data(subjects, init_value=[])
        test_split_ids = prepare_benchmark_data_split(base_path, subjects, avoid_ids=exist_ids, min_subject_size=MIN_SUBJECT_SAMPLE, ratio=0.01)
        with open(test_split_id_path, "w") as file:
            json.dump(test_split_ids, file, indent=4)
        # _ = input("continue>>>")

        exist_ids = test_split_ids
        dev_split_ids = prepare_benchmark_data_split(base_path, subjects, avoid_ids=exist_ids, min_subject_size=MIN_SUBJECT_SAMPLE, ratio=0.01)
        with open(dev_split_id_path, "w") as file:
            json.dump(dev_split_ids, file, indent=4)
        # _ = input("continue>>>")

        exist_ids = concat_categorized_data(subjects, test_split_ids, dev_split_ids)
        train_split_ids = prepare_benchmark_data_split(base_path, subjects, avoid_ids=exist_ids, select=False)
        with open(train_split_id_path, "w") as file:
            json.dump(train_split_ids, file, indent=4)
        # _ = input("continue>>>")


