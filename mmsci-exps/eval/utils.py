import torch
import argparse


def create_prompt(model_name, question, answer=None):
    if model_name == 'blip2':
        prompt = f'Question: {question}\nAnswer:'
    elif model_name == 'llava':
        prompt = f'USER: <image>\n{question}\nASSISTANT:'
    elif model_name == 'kosmos2':
        prompt = f'<grounding> {question}'
    elif model_name == 'llava-next-mistral':
        prompt = f"[INST] <image>\n{question} [/INST]"
    if answer:
        prompt += " " + answer
    return prompt


def postprocess_output(model_name, output):
    if model_name == 'blip2':
        return output
    elif model_name == 'llava':
        # return output.split('ASSISTANT: ')[1].split(': ')[0].strip()
        return output.split('ASSISTANT: ')[1].strip()
    return output

def postprocess_output_new(model_name, output):
    if model_name == 'blip2':
        return output
    elif model_name == 'llava':
        if 'ASSISTANT:' in output:
            tmp = output.split('ASSISTANT: ')
            if len(tmp) > 1:
                return tmp[1].strip()
            return output
        return output
    return output

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
import re

def extract_ABCD_colon(input_string):
    input_string = input_string.replace("\n", " ")
    pattern = r'(^| )([A-D]):'
    match = re.search(pattern, input_string)
    if match:
        return match.group(2)
    return None



if __name__ == "__main__":
    # Example usage
    input_string = "dasd fgsfgs A:This is a test. B: Another test. X: Not a match. C: Yet another test."
    matched = extract_ABCD_colon(input_string)
    print(matched)