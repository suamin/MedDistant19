# MedDistant19

<p align="center">
  <img width="50%" src="https://github.com/suamin/MedDistant19/blob/main/imgs/meddistant19.png" />
</p>

This is the data creation repository for the paper:  **MedDistant19: A Challenging Benchmark for Distantly Supervised Biomedical Relation Extraction**. Check out the [baselines](https://github.com/pminervini/meddistant-baselines) repository as well.

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

*MedDistant19* is a distantly supervised biomedical relation extraction (Bio-DSRE) corpus obtained by aligning the PubMed MEDLINE abstracts from 2019 with the SNOMED-CT knowledge graph (KG), derived from the UMLS Metathesaurus 2019. Lack of benchmark, reproducibility, and other inconsistencies in previous works called for the curation of such a resource, leading to a more challenging benchmark.

## Installation

Please use the requirements file for installation:

```bash
pip install -r requirements.txt
```

## Download

**Before Downloading**: Please make sure you have obtained the UMLS license to make use of this dataset. For more details please read the note [here](https://github.com/suamin/MedDistant19/blob/1bc0f0ebede7387ffa15325e156ab8cf352aa0fd/benchmark/README.md).

```bash
cd benchmark
bash download_meddistant19.sh
```

This will download the data in OpenNRE compatiable format in the directory `benchmark/meddistant19`. An example line looks as follows:

```json
{
    "text": "Urethral stones are rarely formed primarily in the urethra and are usually associated with urethral strictures or diverticula .", 
    "h": {"id": "C0041967", "pos": [51, 58], "name": "urethra"}, 
    "t": {"id": "C0041974", "pos": [91, 110], "name": "urethral strictures"}, 
    "relation": "finding_site_of"
}
```

The text is pre-tokenized with ScispaCy and can be split at whitespace. The position indexes are at character level.

### Statistics

The dataset is constructed using the inductive KG split (see below). The summary statistics of final data is presented in the following table:

| Split     | Instances  | Facts     | Rare (%)  | Bags     | NA (%)  |
| --------- |:----------:|:---------:|:---------:|:--------:|:-------:|
| Train     | 251,558    | 2,366     |  92.3%    | 80,668   | 96.9%   |
| Valid     | 179,393    | 806       |  87.8%    | 31,805   | 98.2%   |
| Test      | 213,602    | 1,138     |  91.3%    | 50,375   | 98.1%   |

As discussed in the paper, the KG split can be inductive or transductive. The table below summarizes both (split ratio: 70%, 10%, 20%):

| Facts             | Train   | Valid  | Test   |
| ----------------- |:-------:|:------:|:------:|
| Inductive (I)     | 345,374 | 62,116 | 130,563|
| Transductive (T)  | 402,522 | 41,491 | 84,414 |

## Create Dataset

### Knowledge Base

We use `UMLS` as our knowledge base with `SNOMED_CT_US` subset-based installation using Metamorphosys. Please note that in order to have reproducible data splits, follow the steps as outlined below.  
  
#### Download and Install UMLS2019AB  

Download [UMLS2019AB](https://download.nlm.nih.gov/umls/kss/2019AB/umls-2019AB-full.zip) and unzip it in a directory (prefer this directory). Set the resulting path of the unzipped directory `umls-2019AB-full`. We will call this path as `UMLS_DOWNLOAD_DIR` in the remaining document.  

##### MetamorphoSys  

Go to `UMLS_DOWNLOAD_DIR/2019AB-full` and use the script `run*` depending on your OS. Once the MetamorphoSys application opens, press the `Install UMLS` button. A window will prompt asking for `Source` and `Destination` paths. The `Source` shall already be set to `UMLS_DOWNLOAD_DIR/2019AB-full`. Create a new folder under `UMLS_DOWNLOAD_DIR` called `MedDistant19` and set it as `Destination` path, it shall look like `UMLS_DOWNLOAD_DIR/MedDistant19`. In the remaining document, these two paths will be called `SOURCE` and `DESTINATION`.  

##### config.prop  

Run the script `init_config.py` to set the path values in the `config.prop` file provided in this directory.  
  
```bash  
python init_config.py --src SOURCE --dst DESTINATION  
```  

Now, use this configuration file in MetamorphoSys for installing the `SNOMED_CT_US` by selecting the `Open Configuration` option.  

##### .RRF Files  

Once UMLS installation is complete with MetamorphoSys, find the `*.RRF` files under the `DESTINATION/META`. Copy `MRREL.RRF`, `MRCONSO.RRF` and `MRSTY.RRF` in this directory.  

##### Semantic Groups File  

Please download the Semantic Groups file from [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt).  Once you have downloaded all the files, please match the resulting MD5 hash values of relevant files as reported in the `mmsys.md5` file in this directory. If you still face mismatches, please report the issue.  

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

This will create several files but the more important ones are `train.tsv`, `dev.tsv` and `test.tsv`. These splits are transductive in nature, i.e., the entities appearing in dev and test sets have appeared in the training set.  
  
##### Inductive Split  

Inductive split refers to the creation of dev and test sets where entities were not seen during training. It uses the files created by transductive split. To create simple inductive split, use:  

```bash
bash scripts/kg_inductive.sh  
```  

## Documents

As our documents we use abstract texts from the PubMed MEDLINE 2019 version available [here](https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2019.html). We provide a processed version of the corpora which has been deduplicated, tokenized and linked to the UMLS concepts with ScipaCy's `UMLSEntityLinker`. You can optionally recreate the corpora by following the steps outlined in the section "From Scratch".

### Download Entity Linked Corpora

Please view the [link](https://drive.google.com/drive/folders/1hZQX_ICNAlMffwCJW3fNnXNVv7g9pGzR?usp=sharing) and download the file `medline_pubmed_2019_entity_linked.tar.gz` (~30GB compressed) in `MEDLINE` folder. Match the md5sum value for the downloaded file. Uncompress the file (~221GB):

```bash
cd MEDLINE
tar -xzvf medline_pubmed_2019_entity_linked.tar.gz
```

This will result extract the file `medline_pubmed_2019_entity_linked.jsonl` where each line is in JSON format with tokenized text with associated UMLS concepts. For example:

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

(Optional Step: Start) This is step is optional if you wish to train your own word2vec model using this corpus. The current default (`word2vec.py`) setup is the one used to obtain the pre-trained PubMed embeddings for word2vec model:

```bash
python word2vec.py --medline_entities_linked_fname MEDLINE/medline_pubmed_2019_entity_linked.jsonl --output_dir w2v_model
```
(Optional Step: End)
 
Assuming that you have already followed the instructions in the UMLS folder, we can create the benchmark splits in OpenNRE format.

The script below creates the the benchmark `med_distant19` with the split files `med_distant19_train.txt`, `med_distant19_dev.txt` and `med_distant19_test.txt` in `MEDLINE` directory:

```bash
bash scripts/create_meddistant19.sh
```

You can move these files to the folder `benchmark`:

```bash
mkdir benchmark/med_distant19
mv ../MEDLINE/med_distant19_*.txt benchmark/med_distant19/
```

Please match the md5 hash values provided in `benchmark/med_distant19`. We can extract several relevant files (semantic types, semantic groups, relation categories etc.) from `benchmark/med_distant19` with:

```bash
python extract_benchmark_metadata.py \
    --benchmark_dir benchmark \
    --umls_dir UMLS \
    --dataset benchmark/med_distant19
```

## From Scratch

### Download Abstracts

First download the abstracts from 2019 and extract the texts from them with script:

```bash
cd ../scripts
sh download_and_extract_abstracts.sh
```

This will produce several `*.xml.gz.txt` files in this directory. 

### Install ScispaCy Model

We use the `en_core_sci_lg` model, please install it first:

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

We ran this command on a cluster with slurm support. It took 9hrs with 32 CPUs (with 4GB memory each) and a batch size of 1024 used internally to make use of spaCy's multi-processing. The number of sentences extracted will be around 151M in the file `MEDLINE/medline_pubmed_2019_sents.txt`. Sort and extract unique sentences:

```bash
cat MEDLINE/medline_pubmed_2019_sents.txt | sort | uniq > MEDLINE/medline_pubmed_2019_unique_sents.txt
```

#### Entity Linking

Previous studies have used exact matching strategies which produce suboptimal concept linking. We use ScispaCy's `UMLSEntityLinker` to extract concepts.

```bash
num_cpus=72
model=en_core_sci_lg
batch_size=4096

# Please set this to a directory with a lot of space! ScispaCy will download indexes the first time which takes space 
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

**WARNING**: This job is memory intensive and requires upto half TB. We ran this command on slurm supported cluster with 72 CPUs (with 6GB memory each) and a batch size of 4096. It took about 40hrs to link about 145M unique sentences.

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{amin2022meddistant19,
  title={MedDistant19: A Challenging Benchmark for Distantly Supervised Biomedical Relation Extraction},
  author={Amin, Saadullah and Minervini, Pasquale and Chang, David and Neumann, G{\"u}nter and Stenetorp, Pontus},
  journal={arXiv preprint arXiv:2204.04779},
  year={2022}
}
```

## Acknowledgement  
  
We are thankful for the transductive split of SNOMED-CT in MedDistant19, which is adopted from [snomed_kge](https://github.com/dchang56/snomed_kge). We are also grateful for the inductive split code by [blp](https://github.com/dfdazac/blp).
