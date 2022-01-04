<p align="center">
  <img width="90%" src="https://https://github.com/suamin/MedDistant19/blob/main/imgs/meddistant19.png" />
</p>

--------------------------------------------------------------------------------

## UMLS-KB  
  
We use `UMLS` as our knowledge base with `SNOMED_CT_US` subset-based installation using Metamorphosys. Please note that in order to have reproducible data splits, follow the steps as outlined below.  
  
### Download and Install UMLS2019AB  
  
Download [UMLS2019AB](https://download.nlm.nih.gov/umls/kss/2019AB/umls-2019AB-full.zip) and unzip it in a directory (prefer this directory). Set the resulting path of the unzipped directory `umls-2019AB-full`. We will call this path as `UMLS_DOWNLOAD_DIR` in the remaining document.  
  
#### MetamorphoSys  
  
Go to `UMLS_DOWNLOAD_DIR/2019AB-full` and use the script `run*` depending on your OS. Once the # MetamorphoSys application opens, press the `Install UMLS` button. A window will prompt asking for `Source` and `Destination` paths. The `Source` shall already be set to `UMLS_DOWNLOAD_DIR/2019AB-full`. Create a new folder under `UMLS_DOWNLOAD_DIR` called `MedDistant19` and set it as `Destination` path, it shall look like `UMLS_DOWNLOAD_DIR/MedDistant19`. In the remaining document, these two paths will be called `SOURCE` and `DESTINATION`.  
  
#### config.prop  
  
Run the script `init_config.py` to set the path values in the `config.prop` file provided in this directory.  
  
```  
python ../init_config.py --src SOURCE --dst DESTINATION  
```  
  
Now, use this configuration file in MetamorphoSys for installing the `SNOMED_CT_US` by selecting the `Open Configuration` option.  
  
#### .RRF Files  
  
Once UMLS installation is complete with MetamorphoSys, find the `*.RRF` files under the `DESTINATION/META`. Copy `MRREL.RRF`, `MRCONSO.RRF` and `MRSTY.RRF` in this directory.  
  
#### Semantic Groups File  
  
Please download the Semantic Groups file from [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt).  
  
Once you have downloaded all the files, please match the resulting MD5 hash values of relevant files as reported in the `mmsys.md5` file in this directory. If you still face mismatches, please report the issue.  
  
### Extract SNOMED-CT Triples  
  
Preprocess UMLS files with the script:  
```bash  
sh ../scripts/umls_preprocess.sh  
```  
### Transductive Split  
Now, extract the relevant triples data:  
```bash  
sh ../scripts/snomed_triples.sh  
```  
This will create several files but the more important ones are `train.tsv`, `dev.tsv` and `test.tsv`. These splits are transductive in nature, i.e., the entities appearing in dev and test sets have appeared in the training set.  
  
### Inductive Splits  
#### A. Inductive Split  
Inductive split refers to the creation of dev and test sets where entities were not seen during training. To create simple inductive split, use:  
```  
sh ../scripts/inductive_split.sh  
```  
This will create split files `ind-train.tsv`, `ind-dev.tsv` and `ind-test.tsv` files.  
#### B. Inductive Split with Definitions  
Next, we will create a second version of inductive split that has definitions available in `UMLS2020AA` kb available from SciSpacy. For this, use the script:  
```  
sh ../scripts/inductive_split_with_def.sh  
```  
This will create split files `ind_def-train.tsv`, `ind_def-dev.tsv` and `ind_def-test.tsv` files.  
  

## PubMed MEDLINE Abstract-Texts

As our texts source we use abstract texts from the PubMed MEDLINE 2019 version available [here](https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2019.html). We provide a processed version of the corpora which has been deduplicated, tokenized and linked to the UMLS concepts with SciSpacy's `UMLSEntityLinker`. You can optionally recreate the corpora by following the steps outlined in the section "From Scratch".

## Download Entity Linked Corpora

Please view the [link](https://drive.google.com/drive/folders/1hZQX_ICNAlMffwCJW3fNnXNVv7g9pGzR?usp=sharing) and download the file `medline_pubmed_2019_entity_linked.tar.gz` (~19GB compressed) in this folder. Match the md5 hash value for the downloaded file. Uncompress the file (~83GB) in this directory:

```bash
tar -xzvf medline_pubmed_2019_entity_linked.tar.gz
```

This will result extract the file `medline_pubmed_2019_entity_linked.jsonl` where each line is in JSON format with tokenized text with associated UMLS concepts. For example:

```
{"text": "A 25 % body surface area , full-thickness scald wound was produced in anesthetized animals .", "mentions": [{"id": "C0005902", "pos": [7, 24], "name": "body surface area"}, {"id": "C2733564", "pos": [27, 47], "name": "full-thickness scald"}, {"id": "C0043250", "pos": [48, 53], "name": "wound"}, {"id": "C1720436", "pos": [70, 82], "name": "anesthetized"}, {"id": "C0003062", "pos": [83, 90], "name": "animals"}]}
```

Assuming that you have already followed the instructions in the UMLS folder, we can create the benchmark splits in OpenNRE format. We create three kind of corpora, transductive, inductive and inductive with definitions. Each benchmark can further be created with different sizes `S` (small), `M` (medium) and `L` (large). 

### Transductive

The example command below creates the the benchmark `med_distant19-L` with the split files `med_distant19-L_train.txt`, `med_distant19-dev_train.txt` and `med_distant19-L_test.txt` in `MEDLINE` directory:

```bash
python ../create_kb_aligned_text_corpora.py \
    --medline_entities_linked_fname ../MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
    --triples_dir ../UMLS \
    --split trans \
    --size L
```

You can move these files to the folder `benchmark`:

```bash
mv ../MEDLINE/med_distant19-L_*.txt benchmark/med_distant19-L/
```

Please match the md5 hash values provided in `benchmark/med_distant19-L`. We can extract several relevant files (semantic types, semantic groups, relation categories etc.) from `benchmark/med_distant19-L` with:

```bash
python ../extract_benchmark_metadata.py \
    --benchmark_dir ../benchmark \
    --umls_dir ../UMLS \
    --dataset ../benchmark/med_distant19-L
```

### Inductive

An example command for the inductive split:

```bash
python ../create_kb_aligned_text_corpora.py \
    --medline_entities_linked_fname ../MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
    --triples_dir ../UMLS \
    --split ind \
    --size L
```

### Inductive with Definitions

An example command for the inductive split with definitions:

```bash
python ../create_kb_aligned_text_corpora.py \
    --medline_entities_linked_fname ../MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
    --triples_dir ../UMLS \
    --split trans \
    --has_def \
    --size L
```

## From Scratch

### Download and Extract PubMed Abstract Texts

First download the abstracts from 2019 and extract the texts from them with script:

```bash
cd ../scripts
sh download_and_extract_abstracts.sh
```

This will produce several `*.xml.gz.txt` files in this directory. 

### Run SciSpacy Tokenization

To extract sentences from the abstract texts, we use SciSpacy for tokenization:

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

### Run SciSpacy Entity Linker

Previous studies have used exact matching strategies which produce suboptimal concept linking. We use SciSpacy's `UMLSEntityLinker` to extract concepts.

```bash
num_cpus=72
model=en_core_sci_lg
batch_size=4096

# Please set this to a directory with a lot of space! SciSpacy will download indexes the first time which takes space 
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

**WARNING** This job is memory intensive and requires upto half TB. We ran this command on slurm supported cluster with 72 CPUs (with 6GB memory each) and a batch size of 4096. It took about 40hrs to link about 145M unique sentences.

## Acknowledgement  
  
If you find UMLS helpful in your work, please cite UMLS paper:  
  
```  
@article{bodenreider2004unified,  
title={The unified medical language system (UMLS): integrating biomedical terminology},  
author={Bodenreider, Olivier},  
journal={Nucleic acids research},  
volume={32},  
number={suppl\_1},  
pages={D267--D270},  
year={2004},  
publisher={Oxford University Press}  
}  
```  
Transductive split of SNOMED-CT is due to [snomed_kge](https://github.com/dchang56/snomed_kge) and citation of their work:  
```  
@inproceedings{chang2020benchmark,  
title={Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings},  
author={Chang, David and Bala{\v{z}}evi{\'c}, Ivana and Allen, Carl and Chawla, Daniel and Brandt, Cynthia and Taylor, Andrew},  
booktitle={Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing},  
pages={167--176},  
year={2020}  
}  
```  
Scripts to generate inductive splits are due to [blp](https://github.com/dfdazac/blp) and citation of their work:  
```  
@inproceedings{daza2021inductive,  
title={Inductive Entity Representations from Text via Link Prediction},  
author={Daza, Daniel and Cochez, Michael and Groth, Paul},  
booktitle={Proceedings of the Web Conference 2021},  
pages={798--808},  
year={2021}  
}  
```