#! /bin/bash

#
#SBATCH --ntasks=100
#SBATCH --job-name=process
#SBATCH --output=slurm-%A.out

for i in `seq 0 1000`;do
    srun -p speech-cpu -N1 -n1 -c1 --exclusive ./preproc/pipeline.sh -d ./data/fswildplus/ -t ChicagoFSWildPlus -j $i -s $1 &
done

wait
