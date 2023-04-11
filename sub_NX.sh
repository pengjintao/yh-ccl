#!/bin/bash
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -t 12:00
#1SBATCH -w ia0407
#1 ia0407   
#rm slurm*
#module purge
#module load mpi/intel/17.0.7-public3
#module load mpi/openmpi/4.1.0-icc20-public3
#module load  mpich/3.1.4_gcc8.1.0-wjl-public3
#module load mpi/mvapich2/network-fenggl-public3
# export PATH=/public3/home/sc53841/yhccl/build/test:$PATH
# procname=reduce
# procname=./build/test/Reducescatter
procname=./build/test/allreduce
# mpiexec -n 32 $procname
mpiexec -n 48 ./build/test/allreduce
# mpiexec -n 48 ./build/test/reducescatter
# mpiexec -n 48 ./build/test/reduce

# for((i=0;i<=10;i++))
# do
#     cat slurm*
# done
# mpirun --mca btl self,vader,openib -np 64 $procname
#mpirun --mca btl self,vader,openib -np 64 bcast
#lscpu
#numactl -H
