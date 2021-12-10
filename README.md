
# MedDistant19


## Introduction

Distant supervision (DS) is an effective method that allows to collect large-scale annotated data. However, the collected annotations are prone to noise. In distantly supervised relation extraction (DSRE), a knowledge graph (KG) is used to align with the textual mentions of entities, and a common approach to tackle the problem is using multi-instance learning (MIL). The benchmark dataset `NYT10` was created by using Freebase aligned with New York Times (NYT) articles from 2010. In this work, we introduce `MedDistant19` a DSRE corpus that is constructed by aligning SNOMED Clinical Terminologies (SNOMED-CT) to the PubMed MEDLINE abstracts from 2019. We find that the existing DSRE corpora for the biomedical domain suffer from severe train-test leakage resulting in inflated performance. The model performance drops by X% by removing the overlap.
