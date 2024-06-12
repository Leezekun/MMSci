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

### Download Vicuna checkpoints (automatically)

We use LLaVA weights for further fine-tuning on our data. The public LLaVA weights are available [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). In our experiments, we specifically use the [LLaVA-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) model.

For detailed instructions on how to train these models, refer to the [LLaVA](https://github.com/haotian-liu/LLaVA) project.


### Data Preparation

Ensure you have prepared the data as instructed in [here](https://github.com/Leezekun/MMSci/blob/main/mmsci-data/README.md).

Specifically, you should have the following in the [mmsci-data/benchmark/train](../mmsci-data/benchmark/train) directory:
- `llava_image_caption_mixed_data.json`: This file stores the conversations.
- `images`: This directory contains all the necessary images.


### Visual Instruction Tuning
For visual instruction tuning, we adhere to the hyperparameters used in in [LLaVA](https://github.com/haotian-liu/LLaVA).

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.6-Vicuna-7B | 128 | 2e-5 | 1 | 2048 | 0 |

Our model is trained on 8 A100 GPUs with 40GB memory. If you're using different GPUs, adjust the `per_device_train_batch_size` and the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

To start training, execute the following commands:
```bash
cd scripts
sh finetune.sh
```

We train for one epoch, which takes approximately 24 hours.

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
python run_captioning.py --model_name $model --k 3 --with_abstract False --with_content False

### Abstract-grounded Figure Captioning
python run_captioning.py --model_name $model --k 3 --with_abstract True --with_content False

### VQA (Image Caption Matching)
python run_vqa.py --model_name $model --k 5 --setting 1
python run_vqa.py --model_name $model --k 5 --setting 2
python run_vqa.py --model_name $model --k 5 --setting 3
```
We provide the scripts for all evaluated open-source models in our experiments in the `./scripts/eval` directory.

3. OpenAI Inferences: For the evaluation of GPT-4V (GPT-4-turbo) and GPT-4o, please refer to the `./eval/prepare_openai_input.ipynb` to prepare the input and perform batch inference, and then `./eval/process_openai_output.ipynb` notebook for process the output. Remember to set your OpenAI api key. We use OpenAI's Batch API to send asynchronous groups of requests with 50% lower costs.

### VQA Evaluation
To evaluate the VQA (image caption matching) performance:
1. Prepare Captioning Output: Ensure that the figure captioning output is available in the `./eval/output/image_caption_matching` directory.
2. Parse answer: For outputs where the answers cannot be easily parsed, run `./eval/parse_vqa_output.py`.
3. Run the Evaluation Notebook: Open and execute the `evaluate_vqa.ipynb` Jupyter notebook. This will process the captioning output and generate the VQA evaluation results.

### Captioning Evaluation
To evaluate the figure captioning performance:
1. Prepare Captioning Output: Ensure that the figure captioning output is available in the `./eval/output/image_caption_generation` directory.
2. Calculate Other Reference-Based Metrics: Run the `textgen_scores.py` script to calculate reference-based metrics such as BLEU, METEOR, and ROUGE. For example, evaluate the outputs of LLaVA-Next-MMSci:
```bash
python textgen_scores.py --model llava-next-mmsci
```
3. Calculate CLIPScore and RefCLIPScore (optional): Execute the `clipscore.py` script to compute the CLIPScore and RefCLIPScore for the generated captions.
4. Review and Print Scores: Open and execute the `print_captioning_scores.ipynb` notebook to print and review the detailed captioning scores.


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): The primary codebase on which our code is based. We use it for our training processes and use its checkpoints.
- [Vicuna](https://github.com/lm-sys/FastChat):  Provides the Vicuna model, which is used as the base language model for the LLaVA model we use.
- [clipscore](https://github.com/jmhessel/clipscore): Used for evaluating CLIPScore and RefCLIPScore

## Licenses
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)

**Usage and License Notices**: This project leverages various checkpoints, and code, each governed by their respective original licenses. Users must adhere to the terms and conditions specified in these licenses. This include but not limited to the Apache 2.0 License for our codebase, as outlined in the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) codebase and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional restrictions beyond those set by the original licenses. Users are responsible for ensuring their use of the datasets and checkpoints complies with all applicable laws and regulations.
