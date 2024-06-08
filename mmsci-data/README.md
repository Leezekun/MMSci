
# Data Downloading for MMSci
We provide the instructions for downloading our data in this `README.md` file. For a detailed description of the dataset, please refer to `DATACARD.md`.

## Source Data

You can directly download the source dataset from [here](https://mmsci.s3.amazonaws.com/rawdata.zip). After downloading, unzip the file to access the `rawdata`.

Alternatively, you can follow the steps below to crawl the data yourself.

### Step 1: Scrape Article Links
First, scrape the links of all available articles:
```bash
mkdir rawdata
cd ./scripts
python scrape_link.py --category all
```
This process will take around 45 minutes. After completion, you will find five directories corresponding to the five major categories in Nature Communications. Each directory contains a text file named `{subject}_article_links.txt`, which records all the available article links for that subject.

### Step 2: Scrape Article Content
Next, run the script the crawl the detailed informaiton of articles from their links:
```bash
cd ./scripts
python scrape_content.py --scrape_pdf False --category all
```
Optionally, you can set --scrape_pdf True if you wish to scrape PDFs.

### Step 3: Process the Source Dataset
Run the preprocessing script to convert math equations expressed in LaTeX format into text format:
```bash
python preprocessing.py
```

## Benchmark Data
We have provided the benchmark constructed from the source data `rawdata` at [here](https://mmsci.s3.amazonaws.com/benchmark.zip).
After downloading and unzipping the file, you will get the `benchmark` directory.

Inside the benchmark directory, you will find three subdirectories:
- `train`: It corresponds to the training set for visual instruction tuning
- `dev`: You can use this set for benchmark evaluation currently.
- `test`: The answers in the test set are anonymized to prevent data contamination. We will provide evaluation scripts to test this set by submitting your output.


## Pretraining Data
Download the interleaved data for visual pre-training at [here](https://mmsci.s3.amazonaws.com/pretraindata.zip).

After downloading and unzipping the file, you will get the `pretraindata` directory. The `./pretraindata/shards` folder contains data formatted for pre-training on interleaved multimodal datasets supported by the [VILA](https://github.com/Efficient-Large-Model/VILA) framework.

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This dataset is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg