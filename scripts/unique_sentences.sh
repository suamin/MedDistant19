#!/bin/sh

cat MEDLINE/medline_pubmed_2019_sents.txt | sort | uniq > MEDLINE/medline_pubmed_2019_unique_sents.txt
