# UMLS-KB  
  
We use `UMLS` as our knowledge base with `SNOMED_CT_US` subset-based installation using Metamorphosys. Please note that in order to have reproducible data splits, follow the steps as outlined below.  
  
## Download and Install UMLS2019AB  
  
Download [UMLS2019AB](https://download.nlm.nih.gov/umls/kss/2019AB/umls-2019AB-full.zip) and unzip it in a directory (prefer this directory). Set the resulting path of the unzipped directory `umls-2019AB-full`. We will call this path as `UMLS_DOWNLOAD_DIR` in the remaining document.  
  
### MetamorphoSys  
  
Go to `UMLS_DOWNLOAD_DIR/2019AB-full` and use the script `run*` depending on your OS. Once the # MetamorphoSys application opens, press the `Install UMLS` button. A window will prompt asking for `Source` and `Destination` paths. The `Source` shall already be set to `UMLS_DOWNLOAD_DIR/2019AB-full`. Create a new folder under `UMLS_DOWNLOAD_DIR` called `MedDistant19` and set it as `Destination` path, it shall look like `UMLS_DOWNLOAD_DIR/MedDistant19`. In the remaining document, these two paths will be called `SOURCE` and `DESTINATION`.  
  
### config.prop  
  
Run the script `init_config.py` to set the path values in the `config.prop` file provided in this directory.  
  
```  
python ../init_config.py --src SOURCE --dst DESTINATION  
```  
  
Now, use this configuration file in MetamorphoSys for installing the `SNOMED_CT_US` by selecting the `Open Configuration` option.  
  
### .RRF Files  
  
Once UMLS installation is complete with MetamorphoSys, find the `*.RRF` files under the `DESTINATION/META`. Copy `MRREL.RRF`, `MRCONSO.RRF` and `MRSTY.RRF` in this directory.  
  
### Semantic Groups File  
  
Please download the Semantic Groups file from [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt).  
  
Once you have downloaded all the files, please match the resulting MD5 hash values of relevant files as reported in the `mmsys.md5` file in this directory. If you still face mismatches, please report the issue.  
  
## Extract SNOMED-CT Triples  
  
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