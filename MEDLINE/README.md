# PubMed MEDLINE Abstract-Texts

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