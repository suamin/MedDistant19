#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=72
MEM_PER_CPU=6GB

# Change anaconda environment
ENV=multirescnn

export python=/netscratch/samin/dev/miniconda3/envs/$ENV/bin/python3.7
export SCISPACY_CACHE=/netscratch/samin/cache/scispacy

model=en_core_sci_lg
batch_size=4096


srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  $python scispacy_entity_linking.py \
  --medline_unique_sents_fname MEDLINE/medline_pubmed_2019_unique_sents.txt \
  --output_file MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
  --scispacy_model_name $model \
  --n_process $NUM_CPUS \
  --batch_size $batch_size \
  --min_sent_tokens 5 \
  --max_sent_tokens 128
