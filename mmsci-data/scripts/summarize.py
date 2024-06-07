import os
import json
from tqdm import tqdm
import argparse
from utils import *
from subjects import subjects
from transformers import AutoTokenizer


def summarize_subject(data_path, subject, tokenizer):
    num_articles, num_figures, num_caption_tokens, \
    num_abstract_tokens, num_article_tokens, num_figure_w_subs, num_subfigures = 0, 0, 0, 0, 0, 0, 0
    earlist_time, latest_time = "2025-01-01", "1500-01-01"

    # iterate each article directory
    subject_path = os.path.join(data_path, subject)
    for entry in tqdm(os.listdir(subject_path)):
        full_path = os.path.join(subject_path, entry)
        if os.path.isdir(full_path):
            uid = os.path.basename(full_path)
            filename = os.path.join(full_path, f"{uid}_data.json")
            with open(filename, "r", encoding='utf-8') as file:
                data = json.load(file)
            
            try:
                # analyze data
                num_figures += len(data["images"])
                for image in data["images"]:
                    caption = image["caption"] + " " + image["description"]
                    if not tokenizer:
                        num_caption_tokens += len(caption.split(" "))
                    else:
                        num_caption_tokens += len(tokenizer.encode(caption))

                    subfigures = find_sub_caption(image["description"])
                    if len(subfigures) > 0:
                        num_figure_w_subs += 1
                    num_subfigures += len(subfigures)

                abstract = data["abstract"]
                if not tokenizer:
                    num_abstract_tokens += len(abstract.split(" "))
                else:
                    num_abstract_tokens += len(tokenizer.encode(abstract))

                sections = data["sections"]
                for section in sections:
                    if not tokenizer:
                        num_article_tokens += len(section["content"].split(" "))
                    else:
                        num_article_tokens += len(tokenizer.encode(section["content"]))

                if "published_time" in data:
                    if convert_date_to_int(data["published_time"]) > convert_date_to_int(latest_time):
                        latest_time = data["published_time"]
                    if convert_date_to_int(data["published_time"]) < convert_date_to_int(earlist_time):
                        earlist_time = data["published_time"]

                num_articles += 1
            
            except:
                continue

    summary = {
        "num_articles": num_articles,
        "num_figures": num_figures,
        "num_figure_w_subs": num_figure_w_subs,
        "num_subfigures": num_subfigures,
        "subfigure_per_figure": num_subfigures / num_figures,
        "figure_per_article": num_figures / num_articles,
        "num_tokens_per_caption": num_caption_tokens / num_figures,
        "num_tokens_per_abstract": num_abstract_tokens / num_articles,
        "num_tokens_per_article": num_article_tokens / num_articles,
        "earlist_time": earlist_time,
        "latest_time": latest_time
    }
    return summary


def summarize_dataset_by_subject(summary):
    dataset_summary = {}
    for category in summary:
        for subject in summary[category]:
            if subject == "Ecology" and category == "Earth and environmental sciences":
                key = "Ecology sciences"
            else:
                key = subject
            dataset_summary[key] = summary[category][subject]
    return dataset_summary


def summarize_dataset(summary):
    num_categories, num_subjects, num_articles, num_figures, num_caption_tokens, \
    num_figure_w_subs, num_subfigures, \
    num_abstract_tokens, num_article_tokens = 0, 0, 0, 0, 0, 0, 0, 0, 0
    earlist_time, latest_time = "2025-01-01", "1500-01-01"

    for category in summary:
        num_categories += 1
        for subject in summary[category]:
            num_subjects += 1
            num_articles += summary[category][subject]["num_articles"]
            num_figures += int(summary[category][subject]["figure_per_article"] * summary[category][subject]["num_articles"])
            num_caption_tokens += int(summary[category][subject]["num_tokens_per_caption"] * summary[category][subject]["num_articles"])
            num_abstract_tokens += int(summary[category][subject]["num_tokens_per_abstract"] * summary[category][subject]["num_articles"])
            num_article_tokens += int(summary[category][subject]["num_tokens_per_article"] * summary[category][subject]["num_articles"])
            num_figure_w_subs += int(summary[category][subject]["num_figure_w_subs"])
            num_subfigures += int(summary[category][subject]["num_subfigures"])
            # compare time
            if convert_date_to_int(summary[category][subject]["latest_time"]) > convert_date_to_int(latest_time):
                latest_time = summary[category][subject]["latest_time"]
            if convert_date_to_int(summary[category][subject]["earlist_time"]) < convert_date_to_int(earlist_time):
                earlist_time = summary[category][subject]["earlist_time"]

    figure_per_article = num_figures / num_articles
    num_tokens_per_caption = num_caption_tokens / num_articles
    num_tokens_per_abstract = num_abstract_tokens / num_articles
    num_tokens_per_article = num_article_tokens / num_articles
    num_subfigure_per_figure = num_subfigures / num_figures

    dataset_summary = {
        "num_categories": num_categories,
        "num_subjects": num_subjects,
        "num_articles": num_articles,
        "num_figures": num_figures,
        "num_figure_w_subs": num_figure_w_subs,
        "num_subfigures": num_subfigures,
        "num_subfigure_per_figure": num_subfigure_per_figure,
        "figure_per_article": figure_per_article,
        "num_tokens_per_caption": num_tokens_per_caption,
        "num_tokens_per_abstract": num_tokens_per_abstract,
        "num_tokens_per_article": num_tokens_per_article,
        "earlist_time": earlist_time,
        "latest_time": latest_time
    }

    return dataset_summary


if __name__ == '__main__':
    
    base_path = "../rawdata"
    
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--category', type=str, default="all") #
    parser.add_argument('--tokenizer', type=str, default=None) # "meta-llama/Llama-2-7b-chat-hf"

    args, unknown = parser.parse_known_args()
    print(args)

    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = None

    summary = {}

    if args.category == "all":
        for category in subjects:
            summary[category] = {}

            data_path = os.path.join(base_path, category)
            save_path = os.path.join(data_path, "summary.json")

            if os.path.exists(save_path):
                with open(save_path, "r") as file:
                    category_summary = json.load(file)
                summary[category] = category_summary[category]

            for subject in subjects[category]:
                if subject not in summary[category]:
                    subject_summary = summarize_subject(data_path, subject, tokenizer)
                    summary[category][subject] = subject_summary

        overall_summary = summarize_dataset(summary)
        summary["Total"] = overall_summary

        save_path = os.path.join(base_path, "summary.json")
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)


    else:
        category = args.category
        summary[category] = {}

        # start analysis
        data_path = os.path.join(base_path, category)
        save_path = os.path.join(data_path, "summary.json")
 
        if os.path.exists(save_path):
            with open(save_path, "r") as file:
                category_summary = json.load(file)
            summary[category] = category_summary[category]

        for subject in subjects[category]:
            if subject not in summary[category]:
                subject_summary = summarize_subject(data_path, subject)
                summary[category][subject] = subject_summary

        overall_summary = summarize_dataset(summary)
        summary["Total"] = overall_summary

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        

                    
                    


