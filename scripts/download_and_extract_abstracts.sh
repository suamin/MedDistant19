#!/bin/sh

NUM_JOBS=16

cd MEDLINE

echo "[1] - Downloading *.xml.gz files ..."
curl https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/2019.html | tr "\"" "\n" | grep ".xml.gz" | grep http | awk '{ print "wget -c " $1 }' | parallel -j $NUM_JOBS

echo "[2] - Extracting texts from *.xml.gz files ..."
ls *.xml.gz | awk '{ print "python ../extract_abstracts.py " $1 " " $1 ".txt" }' | parallel -j $NUM_JOBS

cd ..
