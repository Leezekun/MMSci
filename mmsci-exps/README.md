# MMSci Benchmark Evaluation and Visual Instruction Tuning

## Contents
- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
To set up the environment for this project, you can follow the setup instructions from the [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#install) project, as our environment configuration is similar. Alternatively, you can follow the steps below to set up the environment directly:


1. Install Package
```Shell
conda create -n mmsci python=3.10 -y
conda activate mmsci
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train

### Data Preparation

Ensure you have prepared the data as instructed in [here](../mmsci-data/README.md).

Specifically, you should have the following in the [mmsci-data/benchmark/train](../mmsci-data/benchmark/train) directory:
- `llava_image_caption_mixed_data.json`: This file stores the conversations.
- `images`: This directory contains all the necessary images.

### Supervised Fine-Tuning
We use the [LLAMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repo to train the Qwen2-VL-2B model on our data. Convert the data into the expected format and add the our data into the `dataset_info.json` of the repo.
```json
"mmsci": {
    "file_name": <data_path>,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
}
```

We train for one epoch. Please use the following script for training.

```yaml
### model
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: mmsci 
template: qwen2_vl
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 32

### output
output_dir: saves/qwen2_vl-2b/lora/sft-mmsci-mixed-v2
logging_steps: 100
save_steps: 1000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

## Evaluation

This is for the evaluation of different models on our benchmark test set.
There are two tasks in our benchmark: figure captioning (image caption generation) and Multiple-choice VQA (image captioning matching)

### Inferences

1. Add Models for Evaluation: For evaluation on open-sourced models, first update the load_modal function in model_loader.py to include the models you want to evaluate. 
For example, add models as follows:
```
models = {
        'blip2': 'Salesforce/blip2-opt-2.7b',
        ...
}
```

2. Run Inferences: Use the following scripts to run inferences on the specified tasks. The `--k` parameter indicates the number of inferences per sample. We recommend using 3 for captioning tasks and 5 for VQA tasks.
```bash
devices=1
export CUDA_VISIBLE_DEVICES=$devices

model=blip2

### Ungrounded Figure Captioning
python run_captioning.py --model_name $model --k 1 --with_abstract False --with_content False

### Abstract-grounded Figure Captioning
python run_captioning.py --model_name $model --k 1 --with_abstract True --with_content False

### VQA (Image Caption Matching)
python run_matching.py --model_name $model --k 1 --setting 1 # Fig2Cap
python run_matching.py --model_name $model --k 1 --setting 2 # SubFig2Cap
python run_matching.py --model_name $model --k 1 --setting 3 # SubCap2Fig
```
We provide the scripts for all evaluated open-source models in our experiments in the `./scripts/eval` directory.

3. Proprietary Model Inferences: For the evaluation of proprietary MLLMs, we support APIs for OpenAI, Anthropic's Claude, and Google's Gemini models. Remember to set the API keys.

### Captioning Evaluation
To evaluate the figure captioning performance:
1. Prepare Captioning Output: Ensure that the figure captioning output is available in the `./eval/output/image_caption_generation` directory.
2. Calculate Other Reference-Based Metrics: Run the `textgen_scores.py` script to calculate reference-based metrics such as BLEU, METEOR, and ROUGE. For example, evaluate the outputs of LLaVA-Next-MMSci:
```bash
python textgen_scores.py --model qwen2-vl-2b
```
3. Calculate the **G-Eval score** with the `llm_judge_scores.py` script.
4. Calculate the **FActScore** with the `textgen.py` script.
5. Calculate the **L3Score** with the `l3score.py` script.

<!-- 3. Calculate CLIPScore and RefCLIPScore (optional): Execute the `clipscore.py` script to compute the CLIPScore and RefCLIPScore for the generated captions. -->
<!-- 4. Review and Print Scores: Open and execute the `print_captioning_scores.ipynb` notebook to print and review the detailed captioning scores. -->


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): The primary codebase on which our code is based. We use it for our training processes and use its checkpoints.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use this codebase and its checkpoints in our visual instruction tuning process. 
- [Vicuna](https://github.com/lm-sys/FastChat):  Provides the Vicuna model, which is used as the base language model for the LLaVA model we use.
- [clipscore](https://github.com/jmhessel/clipscore): Used for evaluating CLIPScore and RefCLIPScore
- [l3score](https://github.com/google/spiqa/blob/main/metrics/llmlogscore/llmlogscore.py): Used for evaluating L3Score for captioning.

## Licenses
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)

**Usage and License Notices**: This project leverages various checkpoints, and code, each governed by their respective original licenses. Users must adhere to the terms and conditions specified in these licenses. This include but not limited to the Apache 2.0 License for our codebase, as outlined in the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) codebase and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional restrictions beyond those set by the original licenses. Users are responsible for ensuring their use of the datasets and checkpoints complies with all applicable laws and regulations.
