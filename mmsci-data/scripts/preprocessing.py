import os
import json
from tqdm import tqdm
import re
import copy
import argparse
from utils import *
from subjects import subjects
from pylatexenc.latex2text import LatexNodes2Text


def process_subject(subject_path):
    # iterate each article directory
    for entry in tqdm(os.listdir(subject_path)):
        full_path = os.path.join(subject_path, entry)
        if os.path.isdir(full_path):
            uid = os.path.basename(full_path)
            processed_filename = os.path.join(full_path, f"{uid}_processed_data.json")
            original_filename = os.path.join(full_path, f"{uid}_data.json")
            with open(original_filename, "r", encoding='utf-8') as file:
                original_data = json.load(file)
            processed_data = copy.deepcopy(original_data)

            try:
                # part 1: image captions
                images = processed_data["images"]
                for image in images:
                    caption = image["description"]
                    formulas = find_formula(caption)
                    # LATEX to Text
                    for formula_latex in formulas:
                        formula_text = LatexNodes2Text().latex_to_text(formula_latex)
                        caption = caption.replace(formula_latex, formula_text)

                    image["description"] = caption

                # part 2: abstract
                abstract = processed_data["abstract"]
                formulas = find_formula(abstract)
                # LATEX to Text
                for formula_latex in formulas:
                    formula_text = LatexNodes2Text().latex_to_text(formula_latex)
                    abstract = abstract.replace(formula_latex, formula_text)
                processed_data["abstract"] = abstract

                # part 3: main content
                sections = processed_data["sections"]
                for section in sections:
                    content = section["content"]
                    formulas = find_formula(content)
                    # LATEX to Text
                    for formula_latex in formulas:
                        formula_text = LatexNodes2Text().latex_to_text(formula_latex)
                        content = content.replace(formula_latex, formula_text)

                    section["content"] = content
                
                # save the processed data
                with open(processed_filename, "w", encoding='utf-8') as file:
                    json.dump(processed_data, file, indent=4, ensure_ascii=False)

            except Exception as e:
                print(e)
                continue
    return


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    # arguments for dataset
    parser.add_argument('--category', type=str, default="all") #

    args, unknown = parser.parse_known_args()
    print(args)

    base_path = "../rawdata"
    all_categories = list(subjects.keys())
    if args.category == "all":
        scraped_categories = all_categories
    else:
        assert args.category in all_categories
        scraped_categories = [args.category]
    
    for category in scraped_categories:
        for subject in subjects[category]:
            print(base_path, category, subject)
            data_path = os.path.join(base_path, category, subject)
            process_subject(data_path)