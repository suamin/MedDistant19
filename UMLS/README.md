# UMLS Knowledge Base

We use `UMLS` as our knowledge base with `SNOMED_CT_US` subset-based installation using the Metamorphosys. **NB.** to have reproducible data splits, follow the steps exactly as outlined below.

## Download UMLS2019AB 

Download [UMLS2019AB](https://download.nlm.nih.gov/umls/kss/2019AB/umls-2019AB-full.zip) and unzip it in a directory. Set the resulting path of unzipped directory `umls-2019AB-full`. We will call this path as `UMLS_DOWNLOAD_DIR` in the remaining document.

## Metamorphosys

Go to `UMLS_DOWNLOAD_DIR/2019AB-full` and use the script `run*` depending on your OS. Once the Metamorphosys application opens, press the `Install UMLS` button. A window will prompt asking for `Source` and `Destination` path. The `Source` shall already be set to `UMLS_DOWNLOAD_DIR/2019AB-full`. Create a new folder under `UMLS_DOWNLOAD_DIR` called `MedDistant19` and set it as `Destination` path, it shall look like `UMLS_DOWNLOAD_DIR/MedDistant19`. In the remaining document, these two paths will be called `SOURCE` and `DESTINATION`.

### config.prop

Run the script `init_config.py` to set the path values in `config.prop` file provided in this directory.

```
python init_config.py --src SOURCE --dst DESTINATION
``` 

Now, use this configuration file in Metamorphosys for installing the `SNOMED_CT_US` by selecting the `Open Configuration` option.

## .RRF Files

Once UMLS installation is complete with Metamorphosys, find the `*.RRF` files under the `DESTINATION/META`. Copy `MRREL.RRF`, `MRCONSO.RRF` and `MRSTY.RRF` in this directory.

## Semantic Groups File

Please download the Semantic Groups file from [here](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt).

Once you have downloaded all the files, please match the resulting MD5 hash values of relevant files as reported in `mmsys.md5` file in this directory. If you still face mismatches, please report the issue.

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
