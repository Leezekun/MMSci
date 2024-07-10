# Dataset Card for MMSci

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Intended Use](#intended-use)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Source Data](#source-data) 
  - [Benchmark Data](#benchmark-data)
  - [Pretraining Data](#pretraining-data)
- [Dataset Creation](#dataset-creation)
  - [Initial Data Collection](#initial-data-collection)
  - [Source Language Producers](#source-language-producers)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)

## Dataset Description

### Dataset Summary

**MMSci** is a **multimodal, multi-discipline dataset** containing high-quality, open-access articles published in [Nature Communications journals](https://www.nature.com/ncomms/). This dataset encompasses [five major subjects and spans 72 diverse science disciplines](https://www.nature.com/nature/browse-subjects), primarily within the natural sciences. We have developed a benchmark to evaluate models' comprehension of graduate-level multimodal scientific knowledge across various advanced disciplines. Additionally, we constructed visual instruction-following data for visual instruction tuning and interleaved text and image data for visual pre-training.

### Intended Use

This dataset is used to evaluate and enhance the foundation models' understanding of advanced multimodal scientific knowledges.

### Languages

English

## Dataset Structure

The dataset consists of three parts:

(1) **rawdata**: This is the **source dataset**, comprising articles and figures crawled from Nature Communications. You can download it directly from this URL:

`https://mmsci.s3.amazonaws.com/rawdata.zip`

(2) **benchmark**: This is the **benchmark data**, consisting of test, dev, and training sets constructed from the source dataset. The training set includes visual instruction-following data. You can download it directly from this URL:

`https://mmsci.s3.amazonaws.com/benchmark.zip`

(3) **pretraindata**: This is the interleaved text and image data used for LVLM pre-training, constructed from the source dataset using the same data splits as the benchmark data. You can download it directly from this URL:

`https://mmsci.s3.amazonaws.com/pretraindata.zip`

### Source Data
The source data is stored in the `rawdata` directory, organized as follows:

#### First-Level Category Directories
These represent the five major categories in our dataset:
- Biological sciences
- Earth and environmental sciences
- Health sciences
- Physical sciences
- Scientific community and society

#### Second-Level Subject Directories
Under each major category, there are directories for each subject within that category.

#### Third-Level Article Directories
Within each subject directory, there are multiple directories, each named with a unique article ID, representing individual articles. Inside each article directory, you will find images of all figures in the article and a {uid}_data.json file containing the following information:
- `pdf_link`: the link of pdf of the article (not needed)
- `review_pdf_link`: the link of the pdf of paper review (if available)
- `unique_id`: the unique aricle id, which is also the directory name
- `images` contains the informaiton of the figures in this article. Each image contains:
    - `image_filename`: image filenames under this directory
    - `text_filename`: the text file containing the caption for this figure
    - `caption`: the main caption for this image
    - `description`: the detailed caption, containing sub-captions
- `title`: article title
- `abstract`: article abstract
- `published_time`: article published time
- `sections`: all the sections in the main article content. Each section contains:
    - `section`: section name
    - `content`: the content in this section
- `references`: all the reference articles in this articles. Each reference contains:
    - `idx`: the index of reference in this article
    - `title`: reference article title
    - `link`: the link of the reference article

Here's an example:

textual data:
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


### Benchmark Data

#### Data Splits
The benchmark data is stored in the `benchmark` directory, consisting of train/dev/test splits. For detailed information on data splits, please refer to the paper. 

#### Figure Captioning Data
The data is stored in `image_caption_generation_data.json` within the `./benchmark/train`, `./benchmark/dev`, and `./benchmark/test` directory, which is used for scientific figure captioning task. 

Each instance contains:
- `uid`: the unique article ID of the article from which this sample is built
- `category`: the category of the article from which this sample is built
- `subject`: the subject of the article from which this sample is built
- `abstract`: the abstract of the article from which this sample is built, for abstract-grounded figure captioning
- `content`: the full article content of the article from which this sample is built, for content-grounded figure captioning
- `image`: the image name of this figure
- `caption`: the caption of this figure

Here is an example:
```JSON
{
    "uid": "ncomms13142",
    "category": "Physical sciences",
    "subject": "Materials science",
    "abstract": "The progress in exploiting new electronic materials has been a major driving force in solid-state physics. As a new state of matter, a Weyl semimetal (WSM), in particular a type-II WSM, hosts Weyl fermions as emergent quasiparticles and may harbour novel electrical transport properties...",
    "content": "Gate-tunable negative longitudinal magnetoresistance in the predicted type-II Weyl semimetal WTe2\nThe progress in exploiting new electronic materials has been a major driving force in solid-state physics...",
    "image": "ncomms13142_figure_1.png",
    "caption": "Angle-dependent negative longitudinal MR of thin WTe2. (a) Sample #1 exhibits only negative longitudinal MR at high magnetic fields, which is apparently suppressed at∼3.05°. (b) Sample #2 exhibits a negative longitudinal MR and a positive MR signal at higher magnetic field, which is apparently suppressed at approximately −1.75°..."
}
```

#### Multiple-Choice VQA Data
The data is stored in `image_caption_matching_data.json` within the `./benchmark/train`, `./benchmark/dev`, and `./benchmark/test` directory, which is used for the multiple-choice VQA task. 

The data contains three settings. Each instance in a setting contains:
- `uid`: the unique article ID of the article from which this sample is built
- `category`: the category of the article from which this sample is built
- `subject`: the subject of the article from which this sample is built
- `question`: the question for this sample
- `answer`: answer for this question, A/B/C/D
- `image`: the image name of this figure

Here is an example:
```JSON
{
    "uid": "ncomms7884",
    "category": "Physical sciences",
    "subject": "Materials science",
    "question": "which of the following options best describes the content in sub-figure (c)?\nA: Concentration-dependent (25–500 μM) UV/Vis absorption spectra ofR4·4Cl at 25 °C in water.\nB: Temperature-dependent (2–80 °C) ICD spectra (200 μM) ofR4·4Cl in water.\nC: UV/Vis absorption (solid lines) and normalised fluorescence spectra (excitation: dashed lines, emission: dotted lines) of aqueous solutions ofR4·4Cl (green), stopper1·Cl (red) and dumbbell precursor2·2Cl (blue).\nD: Normalised concentration-dependent (25–500 μM) fluorescence emission spectra (λexcitation=341 nm) ofR4·4Cl at 25 °C in water.",
    "answer": "D",
    "image": "ncomms7884_figure_1.png"
},
```

**Answer Anonymized**: Note that the answers in `./benchmark/test` directory have been anonymized to avoid data contamination. For immediate evaluation, use the dev set. We will provide evaluation scripts to allow you to submit your model's output for official assessment.

#### Visual Instruction-Following Data
The data is stored in `image_caption_chat_data.json` in the `./benchmark/train` directory, which is used for visual instruction tuning. This data is only contained in the train set.

Each instance contains:
- `uid`: the unique article ID of the article from which this sample is built
- `category`: the category of the article from which this sample is built
- `subject`: the subject of the article from which this sample is built
- `image`: the image name of this figure
- `caption`: the caption of this figure
- `conversations`: the conversations regarding this figure

Here is an example:
```JSON
{
    "uid": "ncomms9379",
    "category": "Physical sciences",
    "subject": "Materials science",
    "image": "ncomms9379_figure_4.png",
    "caption": "CPL detector with RH and LH elements patterned into the Vanderbilt University logo.",
    "conversations": [
        {
            "from": "human",
            "value": "Would you please describe the information in sub-figure (d)?"
        },
        {
            "from": "assistant",
            "value": "Scanning photocurrent maps of the metamaterial under LCP (left) and RCP (right) illumination. Scale bar, 10 μm."
        },
        {
            "from": "human",
            "value": "Could you explain what is shown in sub-figure (c)?"
        },
        {
            "from": "assistant",
            "value": "Camera images of the metamaterial under LCP (left) and RCP (right) illumination."
        },
        {
            "from": "human",
            "value": "Overview of sub-figure (a)?"
        },
        {
            "from": "assistant",
            "value": "Schematic of the pattern with the LH metamaterial filling the black region and the RH metamaterial filling the white region."
        },
        {
            "from": "human",
            "value": "What does sub-figure (b) say?"
        },
        {
            "from": "assistant",
            "value": "Camera image of the metamaterial under linearly polarized light with polarization along the vertical direction."
        }
    ]
}
```

### Pretraining Data
The data is stored in the `pretraindata` directory, containing interleaved article text and figure images, formated according to [mmc4](https://github.com/allenai/mmc4). For a detailed description of the data format, please refer to [this documentation](https://github.com/allenai/mmc4/blob/main/DATASET_CARD.md#data-instances).

## Data Creation

### Initial Data Collection

See the paper for more details.

### Source Language Producers

Authors of publicly accessible articles.

### Annotations

#### Annotation process

The dataset does not include explicit annotations. Instead, the authors themselves carried out a small-scale manual review and classification of the image types specifically for analysis. No external annotators or crowdworkers were involved in this process.

#### Who are the annotators?

N/A

### Personal and Sensitive Information

This dataset does not include any personal or sensitive information. All author information is open access, and we do not explicitly extract, store, or use any author information.

## Considerations for Using the Data

### Social Impact of Dataset

Potential benefits:

- **Evaluation Benchmark**: This dataset serves as a valuable evaluation benchmark for assessing the understanding of large vision-language models (LVLMs) regarding scientific articles and figures.
- **Training Resource**: It can be used as a training resource to enhance LVLMs' comprehension of scientific articles and figures, improving their performance in various scientific and research-related tasks.

Potential risks:

- **Misuse in Academic Integrity**: The advancement of AI research assistants facilitated by this dataset could potentially lead to misuse, such as academic fraud, fabrication, or improper assistance in academic work. We strongly encourage users to exercise caution and responsibility when using AI assistants, ensuring they are employed ethically and correctly.

- **Data Misinterpretation and Hallucination**: There is a risk of misinterpreting the dataset's content, leading to inaccurate conclusions or misuse of scientific information. Users should critically assess and validate the AI-generated outputs against established scientific knowledge and principles.

### Other Known Limitations
Currently, the evaluation benchmark focuses primaryly on understanding the figures in scientific articles. We encourage further efforts to build more comprehensive evaluations on scientific knowledge using our dataset.

## Additional Information

### Licensing Information
This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.