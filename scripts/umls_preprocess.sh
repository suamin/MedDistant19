#!/bin/sh

# Start from MRCONSO to extract concept list
# This command filters out inactive, nonpreferred concepts and only keeps relevant columns
awk -F '|' '$5=="PF" && $12=="SNOMEDCT_US" && $13=="PT" && $17=="N" {OFS="\t"; print$1,$15}' UMLS/MRCONSO.RRF > UMLS/active_concepts.txt

# This command filters out inactive relations and only keeps relevant columns
## NB. we reverse $1 and $5 as in UMLS we have (t, r, h), hence make it (h, r, t)
awk -F '|' '$11=="SNOMEDCT_US" && $15=="N" {OFS="\t"; print$5,$4,$1,$8}' UMLS/MRREL.RRF > UMLS/active_relations.txt

# This command keeps only relevant columsn from MRSTY
awk -F '|' '{OFS="\t"; print$1,$2,$4}' UMLS/MRSTY.RRF > UMLS/semantic_types.txt

# Then we use the notebook umls_utils.ipynb to subset the concepts by semantic types/groups using MRSTY and SemGroups.txt, and subset MRREL using those concepts
