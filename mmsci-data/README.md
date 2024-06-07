
# Data Curation for MMSci

## Source Data

You can directly download the source dataset from [here](https://storage.googleapis.com/zekunli/rawdata.zip). After downloading, unzip the file to access the `rawdata`.

Alternatively, you can follow the steps below to crawl and process the data yourself.

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
Optionally, you can set --scrape_pdf True if you wish to scrape PDFs, though this is not necessary.

Upon completion, you will have subject directories within each category directory. Each subject directory contains multiple directories, each corresponding to an article with its unique article ID as the directory name. Inside each article directory, you will find images of all figures in the article and a {uid}_data.json file containing all the textual data:
```json
{
    "pdf_link": "https://www.nature.com/articles/ncomms1240.pdf",
    "review_pdf_link": "URL not found",
    "unique_id": "ncomms1240",
    "images": [
        {
            "image_filename": "figure_0.png",
            "text_filename": "figure_0_info.txt",
            "caption": "Figure 1: 5hmC preferentially appears in the paternal genome of early mouse preimplantation embryos.",
            "description": "(a) Representative images of mid PN3 and metaphase stage zygotes stained with 5-methylcytosine (5mC, green mouse monoclonal, gift from Dirk Schübeler) and 5-hydroxymethylcytosine (5hmC, red rabbit polyclonal from Active Motif) antibodies. (b) Dynamic appearance of 5hmC during early preimplantation development. Shown are representative images of embryos stained with DNA (blue mouse monoclonal from Millipore) and 5hmC (red rabbit polyclonal from Active Motif) antibodies. (c) Quantification of 5hmC signal normalized against DNA antibody signal. A total of 12–18 precisely staged embryos per pronuclear stage from 3 to 5in vitrofertilization (IVF) experiments were analysed..."
        },
        ...
    ],
    "title": "5-Hydroxymethylcytosine in the mammalian zygote is linked with epigenetic reprogramming",
    "published_time": "2011-03-15",
    "abstract": "The epigenomes of early mammalian embryos are extensively reprogrammed to acquire a totipotent developmental potential. A major initial event in this reprogramming is the active loss/demethylation of 5-methylcytosine (5mC) in the zygote. Here, we report on findings that link this active demethylation to molecular mechanisms. We detect 5-hydroxymethylcytosine (5hmC) as a novel modification in mouse, bovine and rabbit zygotes...",
    "sections": [
        {
            "section": "Introduction",
            "content": "Genome-wide reprogramming of DNA methylation (5-methylcytosine; 5mC) is an important epigenomic process observed in mammalian primordial germ cells and early embryos. The reprogramming of DNA methylation has a direct influence on genomic imprints, the regulation of pluripotency and stem cell networks, the erasure of epimutations and the transcriptional control of transposons [1] , [2] , [3] . In the mammalian zygote, the 5mC content of the paternal chromosomes is substantially reduced before replication of DNA, as detected by immunofluorescence (IF) imaging with antibodies against 5mC [4] , [5] , [6] , [7] . Several lines of evidence suggest that the conversion of 5mC to cytosine can be induced by deamination followed by DNA glycosylase-induced base excision repair [8] , [9] , [10] . We recently analysed the dynamics of active loss/demethylation of 5mC using bisulphite sequencing on staged mouse zygotes. We could indeed find a potential link between 5mC demethylation and base excision repair..."
        },
        ...
    ],
    "references": [
        {
            "idx": "1",
            "title": "Reik, W., Dean, W. & Walter, J. Epigenetic reprogramming in mammalian development.Science293, 1089–1093 (2001).",
            "link": "https://doi.org/10.1126%2Fscience.1063443"
        },
        ...
    ]
}
```

### Step 3: Process the Source Dataset
Run the preprocessing script to convert math equations expressed in LaTeX format into text format:
```bash
python preprocessing.py
```

## Benchmark Data
We have provided the benchmark constructed from the source data `rawdata` at [here](https://storage.googleapis.com/zekunli/benchmark.zip).
After downloading and unzipping the file, you will get the `benchmark` directory.


Alternatively, you can follow the steps below to construct the benchmark data yourself.

### Step 1: Data Split (Optional)
Run the prepare_benchmark.py script to split the data (articles) for train/dev/test sets for building benchmarks and visual instruction following data:
```bash
mkdir benchmark
cd ./scripts
python prepare_benchmark.py
```
Note: We have provided the split IDs, so this step can be skipped if desired.

### Step 2: Process Benchmark Test/Dev/Train Sets
Run the process_benchmark.py script to process the data (articles) for train/dev/test sets for building benchmarks and training sets for visual instruction following data:
```bash
python process_benchmark.py
```

### Step 3: Build Visual Instruction Tuning Data from Train Set
Run the process_llava_sft.py script to build visual instruction following data from the train set:
```bash
python process_llava_sft.py
```

## Pretraining Data
We have provided the interleaved data for visual pre-training in the format used by [VILA](https://github.com/Efficient-Large-Model/VILA) at [here](https://storage.googleapis.com/zekunli/pretraindata.zip).
After downloading and unzipping the file, you will get the `pretraindata` directory.

Alternatively, you can run the `process_pretraining.py` below to construct the interleaved data for pre-training yourself.
```bash
mkdir pretraindata
cd ./scripts
python process_pretraining.py
```

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg