import re
import json
import copy
import argparse
import random
import os
from PIL import Image
from pylatexenc.latex2text import LatexNodes2Text

MIN_CAPTION_WORDS = 50
MIN_SUBJECT_SAMPLE = 5
MAX_SUBJECT_SAMPLE = 50

def convert_date_to_int(date_str):
    try:
        # Split the date string into components
        year, month, day = date_str.split('-')
        # Combine the components into a single integer
        date_int = int(year) * 10000 + int(month) * 100 + int(day)
        return date_int
    except:
        print(date_str)
        return 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_image(image_folder, image_file):

    image_path = os.path.join(image_folder, image_file)
    if not os.path.exists(image_path):
        return False
    try:
        i = Image.open(image_path).convert('RGB')
        return True
    except:
        return False

def find_formula(text):
    # pattern = r'(\$\$.*?\$\$)'
    # matches = re.findall(pattern, text)
    # return matches

    # Pattern to detect LaTeX enclosed by $$ or not
    pattern = re.compile(r'(\$\$.*?\$\$|\\\(.*?\\\))')

    # Find all matches for the pattern
    matches = pattern.findall(text)
    
    # Extract the LaTeX content without the enclosing delimiters
    extracted_matches = []
    for match in matches:
        if match.startswith('$') and match.endswith('$'):
            extracted_matches.append(match[2:-2])
        elif match.startswith('\\(') and match.endswith('\\)'):
            extracted_matches.append(match[2:-2])
        else:
            extracted_matches.append(match)
    
    return extracted_matches

def remove_duplicates(input_list):
    return list(dict.fromkeys(input_list))

def convert_to_multichoice(choices, answer):
    keys = ["A", "B", "C", "D"]
    # assert len(choices) == 4

    random.shuffle(choices)

    answer_choice = choices.index(answer)
    answer_choice = keys[answer_choice]

    mc = {}
    for idx, choice in enumerate(choices):
        key = keys[idx]
        mc[key] = choice
    return mc, answer_choice



def find_integer_before_colon(text):
    # Use a regex pattern to find integers followed by a colon or at the end of the string
    pattern = r'(\d+):|(\d+)$'
    matches = re.search(pattern, text)
    if matches:
        # Try to return the first group (digits followed by a colon), if not found, return the second group (digits at the end)
        return int(matches.group(1) if matches.group(1) else matches.group(2))
    else:
        try:
            return int(text.split()[1])
        except:
            pass
    return -1  # Return None if no match is found


def find_sub_caption(text):
    pattern = r'\(([a-z])\)'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return {}

    valid_matches = []
    for idx, match in enumerate(matches):
        # make sure must at the beginning of a sentence
        if match.start() == 0 or \
            (text[match.start() - 2] in '.!?;:' and text[match.start() - 1] == " "):
            valid_matches.append(match)
            
    sections = {}
    for i in range(len(valid_matches) - 1):
        start = valid_matches[i].end()
        end = valid_matches[i + 1].start()
        key = valid_matches[i].group(1)
        section_text = text[start:end].strip()
        if section_text:
            sections[key] = section_text

    if valid_matches:
        last_key = valid_matches[-1].group(1)
        last_section_text = text[valid_matches[-1].end():].strip()
        if last_section_text:
            sections[last_key] = last_section_text

    return sections


def find_caption(text):
    # Define the regular expression pattern to match "x:" or "x  :" where x is an integer
    pattern = r'\b(\d+)\s*:\s*(.*)'

    # Use re.search to find the pattern in the input string
    match = re.search(pattern, text)
    
    # If a match is found, return the part after the pattern, otherwise return the original string
    if match:
        return match.group(2)
    else:
        return text

def print_dict_with_indent(d):
    print(json.dumps(d, indent=4))


def categorized_data(subjects, init_value=""):
    d = {}
    for category in subjects:
        d[category] = {}
        for subject in subjects[category]:
            d[category][subject] = copy.deepcopy(init_value)
            
    return d

def concat_categorized_data(subjects, c1, c2):
    new_c = {}
    for category in subjects:
        new_c[category] = {}
        for subject in subjects[category]:
            new_c[category][subject] = copy.deepcopy(c1[category][subject]) + \
                                        copy.deepcopy(c2[category][subject])
    return new_c


if __name__ == "__main__":

    caption = "In all spectra, 'a.u.' represents arbitrary units. (a) Chemical structure of the CoTPP molecule (top) and schematic view of the XMCD experiment (bottom). (b) Chemical identification of Co and Ni: L-edges X-ray absorption spectra of Co (CoTPP, photon energy range: 765–815 eV) and Ni (substrate, photon energy range: 835–885 eV) acquired with circularly polarized X-ray light from a synchrotron source with opposite helicities (μ+and μ−). The difference in X-ray absorption for the opposite helicities (dichroism) reveals the magnetization of the observed chemical species. (c) Spin-switching sequence from left to right as indicated by arrows: L-edges XMCD spectra of Co (top panels) and Ni (bottom panels) recorded on the CoTPP/Ni(001) system after the initial preparation of molecular adlayers (left), after NO addition (centre) and on temperature-induced NO desorption (right). The directions of the remanent substrate magnetization M are indicated by grey arrows to the left of each spectrum. Ferromagnetic ordering of molecular spins with respect to the substrate is observed initially. Reversible 'off–on' switching of Co magnetization is observed with progressing NO addition and temperature-induced NO desorption."
    print(find_sub_caption(caption))

    # strings = [
    #     "Figure 2: xasdasfas",
    #     "Fig 5: dasdasda",
    #     "Fig. 1: dasdasdasd",
    #     "fig 1 : idagfadfg",
    #     "No figure here"
    # ]

    # for s in strings:
    #     result = find_caption(s)
    #     print(f"Result: {result}")

    caption = r"Fig. 3: (Type 2 GRS) from the WTCCC dataset:\({{{{{{{\rm{T}}}}}}}}1{{{{{{{\rm{D}}}}}}}}\left(n=1,963\right),{{{{{{{\rm{T}}}}}}}}2{{{{{{{\rm{D}}}}}}}}(n=1,924)\)."
    caption = r"Fig 2: $\({{{{{{{\rm{T}}}}}}}}1{{{{{{{\rm{D}}}}}}}}\left(n=1,963\right),{{{{{{{\rm{T}}}}}}}}2{{{{{{{\rm{D}}}}}}}}(n=1,924)\)$"

    # LATEX to Text
    # Find LaTeX formulas
    formulas = find_formula(caption)
    print(formulas)

    # Initialize the converted caption as an empty string
    converted_caption = caption

    # Process and replace each LaTeX formula
    for formula_latex in formulas:
        formula_text = LatexNodes2Text().latex_to_text(formula_latex)
        print(f"Original: {formula_latex}")
        print(f"Converted Text: {formula_text}")
        # _ = input("continue>>>")

        # Replace the LaTeX formula with its plain text version in the caption
        converted_caption = converted_caption.replace(formula_latex, formula_text, 1)

    # Print the final caption
    print("Original Caption:", caption)
    print("Converted Caption:", converted_caption)

    # with open('captions.txt', 'w') as file:
    #     file.write("Original Caption:\n")
    #     file.write(caption + "\n\n")
    #     file.write("Converted Caption:\n")
    #     file.write(converted_caption + "\n")

    # formula_latex = "\({{{{{{{\rm{T}}}}}}}}1{{{{{{{\rm{D}}}}}}}}\left(n=1,963\right),{{{{{{{\rm{T}}}}}}}}2{{{{{{{\rm{D}}}}}}}}(n=1,924)\)"
    # formula_latex = "{{{{m{D}}}}}}}}(n=1,924)"
    # formula_text = LatexNodes2Text().latex_to_text(formula_latex)
    # print(formula_text)
    # conversation = Conversation()
    # print([x.format("c") for x in conversation.get_conv_template(10, 0.6)])