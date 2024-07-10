# MMSci
<p align="center">
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<img src='https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg'>
</p>

This repo contains all the data and code related to the paper **MMSci: A Multimodal Multi-discipline Dataset for Graduate-Level Scientific Comprehension**

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Benchmark Evaluation & Visual Instruction Tuning](#benchmark)
- [Pre-training on Interleaved data](#pretraining)
- [Materials Generation](#matgen)
- [Resources](#resources)

## Overview
The code and experiments of this project can be structured into four main parts:
1. **Dataset**: Contains all the necessary files for dataset download, collection, and processing. This can be found in the [mmsci-data](mmsci-data) directory.
2. **Benchmark Evaluation & Visual Instruction Tuning**: Involves the creation of benchmark data and visual instruction tuning. Instructions and scripts are available in the [mmsci-exps](mmsci-exps) directory.
3. **Pre-training on Interleaved Data**: Focuses on pre-training the LLaMA2-7B model using our interleaved multimodal dataset. 
4. **Material Generation**: Evaluates the LLaMA2-7B model pre-trained on our data on the task of material generation. 

We put this codebase under the `/home/ubuntu` directory, specifically at `/home/ubuntu/MMSci`. Please replace this path with the path where you have placed this code on your system.

## Dataset
The mmsci-data directory contains all the necessary data for benchmark evaluation, visual instruction tuning, and pre-training on interleaved data. For detailed information, refer to the [mmsci-data/README.md](./mmsci-data/README.md).
 - **Data Card**: Comprehensive details about our dataset can be found in the  [mmsci-data/DATACARD.md](./mmsci-data/DATACARD.md)
 - **License**: Review the licensing terms for our dataset at [mmsci-data/LICENSE](./mmsci-data/LICENSE)

Ensure that the data preparation step is completed before proceeding with any experiments. Ensure that you have prepared the following data files in their respective locations:
 - **rawdata**: This is the source dataset containing all articles and associated figures.
 - **benchmark**: Includes the test/dev sets for benchmark evaluations and the training data for visual instruction tuning.
 - **pretraindata**: Contains the interleaved data necessary for pre-training the model in the Pre-training phase.

## Benchmark Evaluation & Visual Instruction Tuning
Once the dataset is ready, head over to the [mmsci-exps](mmsci-exps) directory for instructions on performing visual instruction tuning and benchmark evaluations. 

Detailed guidelines are provided in the [mmsci-exps/README.md](./mmsci-exps/README.md).

## Pre-training on Interleaved data
In the pre-training phase, we use our prepared interleaved data in `mmsci-data/pretraindata` to continue pre-training a LLaMA2-7B model. 

### Setup VILA
We use the codebase of [VILA](https://github.com/Efficient-Large-Model/VILA) for pre-training vision language models on interleaved data.

Clone the VILA environment and switch to the version we use as follows:
```bash
git clone https://github.com/Efficient-Large-Model/VILA.git
cd VILA
git checkout eaadb1e55a088978ce06abb6242edc251fb4665a
```
Follow the environment setup and data preparation instructions provided in the VILA project.

### Register Our Data MMSci
Ensure the data in `mmsci-data/pretraindata/shards` has been prepared in the Dataset phase, and move it to `VILA/playground/data/mmsci`. 

Then, modify the `datasets_mixture.py` file in the `VILA/llava/data` directory by locating the `register_datasets_mixtures` function and adding the following code to register the MMSci dataset:
```python
mmsci = Dataset(
        dataset_name='mmsci',
        dataset_type='mmc4',
        data_path='./playground/data/mmsci/all')
add_dataset(mmsci)
```
The MMSci data is organized in the same format with MMC4.

Then, add this line at the end of the code:
```python
DATASETS_MIXTURES.update({'mmc4core_mmsci': [mmc4core,mmsci]})
```

### Pre-training
After setting up the environment and registering the MMSci dataset, you can proceed with the pre-training of the model. The pre-training process in VILA involves two main stages.

#### Stage 1: Alignment
To align the textual and visual modalities, move the following script [resources/2_pre-train_mmc4_mmsci.sh](resources/2_pre-train_mmc4_mmsci.sh) into the VILA directory and run it. The [LLaVA-CC3M-pre-train-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) dataset is used for this process. Execute the alignment script with the following command:
```bash
bash 1_mm_align.sh [BASE_MODEL_PATH] [OUTPUT_NAME]
```

In our experiments, we set `BASE_MODEL_PATH` to the path of the base model, which is `meta-llama/Llama-2-7b-hf`. We use `llama2-7b-mm-align-mlp2x`as the `OUTPUT_NAME` to save the aligned model. Therefore, the command becomes:
```bash
bash 1_mm_align.sh meta-llama/Llama-2-7b-hf ./checkpoints/llama2-7b-mm-align-mlp2x
```

#### Stage 2: Pre-training
We have prepared a script for pre-training the model using our data, located at [resources/2_pre-train_mmc4_mmsci.sh](resources/2_pre-train_mmc4_mmsci.sh). To initiate the pre-training process, move the script in the VILA codebase and execute it with the following command:
```bash
bash 2_pre-train_mmc4_mmsci.sh [CODE_PATH] [BASE_MODEL_PATH] [STAGE1_PATH] [OUTPUT_NAME]
```
`CODE_PATH` is the absolute path to the VILA codebase, `BASE_MODEL_PATH` has similar meaning to what is presented in the alignment stage script, which is `meta-llama/Llama-2-7b-hf` in our experiments. `STAGE1_PATH` points to the OUTPUT_NAME of stage 1 (i.e. where the stage 1 checkpoint is stored), which is `llama2-7b-mm-align-mlp2x` in our case. `OUTPUT_NAME` is the desired folder name under checkpoints that saves the pre-training checkpoint. We use `llama2-7b-mmsci` in our case. The trained model is then saved at `VILA/checkpoints/llama2-7b-mmsci`. Therefore, the command becomes:
```bash
bash 2_pre-train_mmc4_mmsci.sh /home/ubuntu/MMSci/VILA meta-llama/Llama-2-7b-hf ./checkpoints/llama2-7b-mm-align-mlp2x ./checkpoints/llama2-7b-mmsci
```


## Materials Generation
In this phase, we use the pre-trained model from the previous pre-training phase as the base model for fine-tuning on material generation tasks. For this, we utilize the [crystal-text-llm](https://github.com/facebookresearch/crystal-text-llm) codebase.

### Install
First, clone the crystal-text-llm repository and navigate to its directory:
```bash
git clone https://github.com/facebookresearch/crystal-text-llm.git
cd crystal-text-llm
```
Follow the setup instructions in the crystal-text-llm repository to configure the environment and prepare the data. You can refer to the detailed [installation guide](https://github.com/Efficient-Large-Model/VILA#installation).

### Fine-tuning
Next, fine-tune the pre-trained model saved in `VILA/checkpoints/llama2-7b-mmsci` for material generation. Use the following command to initiate fine-tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python llama_finetune.py \
                        --run-name llama2-7b-mmsci \
                        --model_name ../VILA/checkpoints/llama2-7b-mmsci \
                        --batch-size 1 \
                        --num-epochs 1 \
                        --fp8
```

### Sampling
After fine-tuning, generate samples using the fine-tuned model with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python llama_sample.py \
                        --model_name llama2-7b-mmsci \
                        --temperature 0.7 \
                        --top_p 0.7 \
                        --batch_size 32 \
                        --num_samples 10000 \
                        --model_name ../VILA/checkpoints/llama2-7b-mmsci \
                        --model_path ./exp/llama2-7b-mmsci/checkpoint-27000 \
                        --out_path ./saved_samples/llama-2-7B-MMSci_0.7_0.7.csv
```
We provided the generated samples by our model in `./resources/llama-2-7B-MMSci_0.7_0.7.csv`.

### Evaluation
Finally, evaluate the generated materials using the following script:
```
python basic_eval.py \
        --model_name llama2-7b-mmsci \
        --samples_path ./saved_samples/llama2-7b-mmsci_0.7_0.7.csv
```

## Resources
We provide various downloadable resources for our MMSci project. Below is a list of the available resources and their corresponding download links:
1. **mmsci-data**: this includes all the data you can download regarding our MMSci data, including: 
  - **rawdata**: The raw (source) data of our dataset can be download [here](https://mmsci.s3.amazonaws.com/rawdata.zip). 
  - **benchmark**: The benchmark dataset, including the data for visual instruction tuning, is available for download [here](https://mmsci.s3.amazonaws.com/benchmark.zip). 
  - **pretraindata**: Interleaved data formatted for pre-training on multimodal datasets can be downloaded [here](https://mmsci.s3.amazonaws.com/pretraindata.zip).

2. **checkpoints**: 
  - **LLaVA-Next-MMSci**: The LLaVA-Next model fine-tuned on our visual instruction-following data is available [here](https://mmsci.s3.amazonaws.com/checkpoints.zip).

:sos: **If you are unable to download the data using the provided link, please try using different browsers.**

## Acknowledgement
We gratefully acknowledge the following projects and codebases that have significantly contributed to our work:
- [LLaVA](https://github.com/haotian-liu/LLaVA): We use this codebase and its checkpoints in our visual instruction tuning process. 
- [VILA](https://github.com/Efficient-Large-Model/VILA): The codebase served as the foundation for our pre-training on interleaved multimodal data. 
- [crystal-text-llm](https://github.com/facebookresearch/crystal-text-llm): We leveraged this codebase for conducting experiments related to material generation.

## Licenses
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)

**Usage and License Notices**: This project incorporates various data, checkpoints, and codebases, each governed by their respective licenses. Users are required to adhere to the terms and conditions outlined in these licenses. Key licenses include:
- **Codebase License**: The primary codebase for our project is licensed under the Apache 2.0 License.
- **Data License**: Our dataset is licensed under the CC BY 4.0 license, which allows for sharing and adaptation with proper attribution. 
