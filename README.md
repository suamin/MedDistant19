# MedDistant19

<p align="center">
  <img width="50%" src="https://github.com/suamin/MedDistant19/blob/main/imgs/meddistant19.png" />
</p>

The data creation repository for the paper:  **MedDistant19: Towards an Accurate Benchmark for Broad-Coverage Biomedical Relation Extraction**. Check out the [baselines](https://github.com/pminervini/meddistant-baselines) repository as well.

--------------------------------------------------------------------------------

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Download](#download)
  - [Statistics](#statistics)
- [Create Dataset](#create-dataset)
  - [KB](#knowledge-base)
  - [Documents](#documents)
- [From Scratch](#from-scratch)
  - [Download Abstracts](#download-abstracts)
  - [Tokenization](#tokenization)
  - [Entity Linking](#entity-linking)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)

## Overview

*MedDistant19* is a distantly supervised biomedical relation extraction (Bio-DSRE) corpus obtained by aligning the PubMed MEDLINE abstracts from 2019 with the SNOMED-CT knowledge graph (KG) derived from the UMLS Metathesaurus 2019.

## Installation

Please use the requirements file for installation:

```bash
pip install -r requirements.txt
```

## Download

**Before Downloading**: Ensure a copy of UMLS license to use this dataset. For more details, please read the note [here](https://github.com/suamin/MedDistant19/benchmark/README.md).

```bash
cd benchmark
bash download_meddistant19.sh
```

This will download the data in [OpenNRE](https://github.com/thunlp/OpenNRE) compatiable format in the directory `benchmark/meddistant19`. An example line looks as follows:

```json
{
    "text": "Urethral stones are rarely formed primarily in the urethra and are usually associated with urethral strictures or diverticula .", 
    "h": {"id": "C0041967", "pos": [51, 58], "name": "urethra"}, 
    "t": {"id": "C0041974", "pos": [91, 110], "name": "urethral strictures"}, 
    "relation": "finding_site_of"
}
```

The text is pre-tokenized with [ScispaCy](https://github.com/allenai/scispacy) and can be split at whitespace. The position indexes are at the character level.

### Statistics

The dataset is constructed using the inductive KG split (see below). The summary statistics of the final data are presented in the following table:

| Split     | Instances  | Facts     | Inst. Per Bag | Bags     | NA (%)  |
| --------- |:----------:|:---------:|:-------------:|:--------:|:-------:|
| Train     | 450,071    | 5,455     |  5.06         | 88,861   | 90.0%   |
| Valid     | 39,434     | 842       |  3.76         | 10,475   | 91.2%   |
| Test      | 91,568     | 1,663     |  4.05         | 22,606   | 91.1%   |

The KG split can be inductive or transductive. The table below summarizes both (split ratio: `70%`, `10%`, `20%`):

| Facts             | Train   | Valid  | Test   |
| ----------------- |:-------:|:------:|:------:|
| Inductive (I)     | 261,797 | 48,641 | 97,861 |
| Transductive (T)  | 318,524 | 28,370 | 56,812 |

## Create Dataset

### Knowledge Base

We use `UMLS` as our knowledge base with `SNOMED_CT_US` subset-based installation using Metamorphosys. Please note that to have reproducible data splits, follow the steps outlined below.  
  
#### Download and Install UMLS2019AB  

Download [UMLS2019AB](https://download.nlm.nih.gov/umls/kss/2019AB/umls-2019AB-full.zip) and unzip it in a directory (prefer this directory). Set the resulting path of the unzipped directory `umls-2019AB-full`. We will call this path `UMLS_DOWNLOAD_DIR` in the remaining document.  

##### MetamorphoSys  

Go to `UMLS_DOWNLOAD_DIR/2019AB-full` and use the script `run*` depending on the OS. Once the MetamorphoSys application opens, press the `Install UMLS` button. A window will prompt asking for `Source` and `Destination` paths. The `Source` shall already be set to `UMLS_DOWNLOAD_DIR/2019AB-full`. Create a new folder under `UMLS_DOWNLOAD_DIR` called `MedDistant19` and set it as `Destination` path, it shall look like `UMLS_DOWNLOAD_DIR/MedDistant19`. In the remaining document, these two paths will be called `SOURCE` and `DESTINATION`.  

##### config.prop  

Run the script `init_config.py` to set the path values in the `config.prop` file provided in this directory.  
  
```bash  
python init_config.py --src SOURCE --dst DESTINATION  
```  

Now, use this configuration file in MetamorphoSys for installing the `SNOMED_CT_US` by selecting the `Open Configuration` option.  

##### .RRF Files  

Once UMLS installation is complete with MetamorphoSys, find the `*.RRF` files under the `DESTINATION/META`. Copy `MRREL.RRF`, `MRCONSO.RRF` and `MRSTY.RRF` in this directory.  

##### Semantic Groups File  

Please download the Semantic Groups file from [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt). Once downloaded all the files, please match the resulting MD5 hash values of relevant files as reported in the `mmsys.md5` file in this directory. If there still are mismatches, please report the issue.  

#### Extract SNOMED-CT KG  

First we will preprocess the UMLS files with the script:

```bash  
bash scripts/preprocess_umls.sh 
```  
##### Transductive Split  

Now, we can extract the transductive triples split:  

```bash  
bash scripts/kg_transductive.sh  
```  

This will create several files, but the more important ones are `train.tsv`, `dev.tsv` and `test.tsv`. These splits are transductive, i.e., the entities appearing in dev and test sets have appeared in the training set.  
  
##### Inductive Split  

Inductive split refers to the creation of dev and test sets where entities were not seen during training. It uses the files created by the transductive split. To create a simple inductive split, use:

```bash
bash scripts/kg_inductive.sh  
```  

## Documents

As our documents, we use abstract texts from the PubMed MEDLINE 2019 version available [here](https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2019.html). We provide a processed version of the corpora, which has been deduplicated, tokenized, and linked to the UMLS concepts with ScipaCy's `UMLSEntityLinker`. We can optionally recreate the corpora by following the steps outlined in the "From Scratch" section.

### Download Entity Linked Corpora

Please view the [link](https://drive.google.com/drive/folders/1hZQX_ICNAlMffwCJW3fNnXNVv7g9pGzR?usp=sharing) and download the file `medline_pubmed_2019_entity_linked.tar.gz` (~30GB compressed) in `MEDLINE` folder. Match the md5sum value for the downloaded file. Uncompress the file (~221GB):

```bash
cd MEDLINE
tar -xzvf medline_pubmed_2019_entity_linked.tar.gz
```

This will result extract the file `medline_pubmed_2019_entity_linked.jsonl`, where each line is in JSON format with tokenized text with associated UMLS concepts and linguistic features per token. For example:

```json
{
  "text": "30 % of ertapenem is cleared by a session of haemodialysis ( HD ) .", 
  "mentions": [
    ["C1120106", "1.00", [8, 17]], 
    ["C1883016", "1.00", [34, 41]], 
    ["C0019004", "1.00", [45, 58]], 
    ["C0019829", "1.00", [61, 63]]
  ], 
  "features": [
    ["CD", "NUM", "NumType=Card", "nummod", 1], 
    ["NN", "NOUN", "Number=Sing", "nsubjpass", 5], 
    ["IN", "ADP", "", "case", 3], 
    ["NN", "NOUN", "Number=Sing", "nmod", 1], 
    ["VBZ", "VERB", "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", "auxpass", 5], 
    ["VBN", "VERB", "Aspect=Perf|Tense=Past|VerbForm=Part", "ROOT", 5], 
    ["IN", "ADP", "", "case", 8], 
    ["DT", "DET", "Definite=Ind|PronType=Art", "det", 8], 
    ["NN", "NOUN", "Number=Sing", "nmod", 5], 
    ["IN", "ADP", "", "case", 10], 
    ["NN", "NOUN", "Number=Sing", "nmod", 8], 
    ["-LRB-", "PUNCT", "PunctSide=Ini|PunctType=Brck", "punct", 12], 
    ["NN", "NOUN", "Number=Sing", "appos", 10], 
    ["-RRB-", "PUNCT", "PunctSide=Fin|PunctType=Brck", "punct", 12], 
    [".", "PUNCT", "PunctType=Peri", "punct", 5]
  ]
}
```

#### Train Word2Vec (Optional)

This step is optional if we wish to train our word2vec model using this corpus. The current default (`word2vec.py`) setup is the one used to obtain the pre-trained PubMed embeddings for the word2vec model:

```bash
python word2vec.py --medline_entities_linked_fname MEDLINE/medline_pubmed_2019_entity_linked.jsonl --output_dir w2v_model
```
(Optional step ends here)
 
Assuming we followed the instructions in the UMLS folder, we can now create the benchmark splits in OpenNRE format.

The script below creates the benchmark `med_distant19` with the split files `med_distant19_train.txt`, `med_distant19_dev.txt`, and `med_distant19_test.txt` in `MEDLINE` directory:

```bash
bash scripts/create_meddistant19.sh
```

We can move these files to the folder `benchmark`:

```bash
mkdir benchmark/med_distant19
mv ../MEDLINE/med_distant19_*.txt benchmark/med_distant19/
```

Please match the md5 hash values provided in `benchmark/med_distant19`. We can extract several relevant files (semantic types, semantic groups, relation categories, etc.) from `benchmark/med_distant19` with:

```bash
python extract_benchmark_metadata.py \
    --benchmark_dir benchmark \
    --umls_dir UMLS \
    --dataset benchmark/med_distant19
```

## From Scratch

### Download Abstracts

First, download the abstracts from 2019 and extract the texts from them with the script:

```bash
cd ../scripts
sh download_and_extract_abstracts.sh
```

This will produce several `*.xml.gz.txt` files in this directory. 

### Install ScispaCy Model

We use the `en_core_sci_lg` model; please install it first:

```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
```

#### Tokenization

To extract sentences from the abstract texts, we use ScispaCy for tokenization:

```bash
model=en_core_sci_lg
num_cpus=32
batch_size=1024

python scispacy_tokenization.py \
    --data_dir MEDLINE \
    --scispacy_model_name $model \
    --n_process $num_cpus \
    --batch_size $batch_size
```

We ran this command on a cluster with SLURM support. It took 9hrs with 32 CPUs (with 4GB memory each) and a batch size of 1024 used internally to use spaCy's multi-processing. The number of extracted sentences will be 151M in the file `MEDLINE/medline_pubmed_2019_sents.txt`. Sort and extract unique sentences:

```bash
cat MEDLINE/medline_pubmed_2019_sents.txt | sort | uniq > MEDLINE/medline_pubmed_2019_unique_sents.txt
```

#### Entity Linking

Previous studies have used exact matching strategies, which produce suboptimal concept linking. We use ScispaCy's `UMLSEntityLinker` to extract concepts.

```bash
num_cpus=32
model=en_core_sci_lg
batch_size=1024

# Please set this to a directory with much space! ScispaCy will download indexes the first time, which takes space 
# export SCISPACY_CACHE=/to/cache/scispacy

python scispacy_entity_linking.py \
    --medline_unique_sents_fname MEDLINE/medline_pubmed_2019_unique_sents.txt \
    --output_file MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
    --scispacy_model_name $model \
    --n_process $num_cpus \
    --batch_size $batch_size \
    --min_sent_tokens 5 \
    --max_sent_tokens 128
```

**WARNING**: This job is memory intensive and requires up to half TB. We ran this command on SLURM supported cluster with 32 CPUs (with ~18GB memory each) and a batch size of 1024. It took about 75hrs to link about 149M unique sentences.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{amin-etal-2022-meddistant19,
    title = "{M}ed{D}istant19: Towards an Accurate Benchmark for Broad-Coverage Biomedical Relation Extraction",
    author = "Amin, Saadullah and Minervini, Pasquale and Chang, David and Stenetorp, Pontus and Neumann, G{\"u}nter",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.198",
    pages = "2259--2277",
}
```

## Acknowledgement  

We thank the original authors of the following sources for releasing their split codes. The transductive split is adopted from [snomed_kge](https://github.com/dchang56/snomed_kge). The inductive split is adopted from [blp](https://github.com/dfdazac/blp).
