#!/bin/sh

# SLURM environment arguments
IMAGE=/netscratch/enroot/dlcc_pytorch_20.07.sqsh
NUM_CPUS=32
MEM_PER_CPU=4GB

# Change anaconda environment
ENV=multirescnn

export python=/netscratch/samin/dev/miniconda3/envs/$ENV/bin/python3.7

data_dir=MEDLINE
model=en_core_sci_lg
batch_size=1024


srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --container-image=$IMAGE \
  --cpus-per-task=$NUM_CPUS \
  --mem-per-cpu=$MEM_PER_CPU \
  --nodes=1 \
  $python scispacy_tokenization.py \
  --data_dir $data_dir \
  --scispacy_model_name $model \
  --n_process $NUM_CPUS \
  --batch_size $batch_size
