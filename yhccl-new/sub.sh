#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 2:00
#SBATCH -w m1401
source /public3/soft/modules/module.sh   
#rm slurm*
#module purge
#module load mpi/intel/17.0.7-public3
#module load mpi/openmpi/4.1.0-icc20-public3
#module load  mpich/3.1.4_gcc8.1.0-wjl-public3
#module load mpi/mvapich2/network-fenggl-public3
export PATH=/public3/home/sc53841/yhccl/build/test:$PATH
#procname=reduce
procname=allreduce
mpirun -np 64 $procname
#mpirun --mca btl self,vader,openib -np 64 $procname
#mpirun --mca btl self,vader,openib -np 64 bcast
#lscpu
#numactl -H
