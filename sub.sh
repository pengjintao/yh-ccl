#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 25:00
#1SBATCH -w  m2501
# source /public3/soft/modules/module.sh   ec1204 ed0701
#rm slurm*
#module purge
#module load mpi/intel/17.0.7-public3
#module load mpi/openmpi/4.1.0-icc20-public3
#module load  mpich/3.1.4_gcc8.1.0-wjl-public3
#module load mpi/mvapich2/network-fenggl-public3
# export PATH=/public3/home/sc53841/yhccl/build/test:$PATH
# procname=reduce
# procname=./build/test/Reducescatter


# ./ipdps18
# export MV2_USE_SHMEM_COLL=1
# export MV2_ENABLE_TOPO_AWARE_COLLECTIVES=1
# export MV2_USE_TOPO_AWARE_ALLREDUCE=1
# export MV2_ENABLE_SOCKET_AWARE_COLLECTIVES=1
# procname=./build/test/bcast
# mpiexec -n 64 $procname
procname=./build/test/allreduce
flag=""
for ((i=0;i<1;i++))
do
# mpiexec $flag   -n 2 $procname
# ./ipdps18 2
# mpiexec $flag  -n 4 $procname
# ./ipdps18 4
# mpiexec $flag  -n 8 $procname
# ./ipdps18 8
mpiexec $flag  -n 16 $procname
./ipdps18 16
mpiexec $flag  -n 32 $procname
./ipdps18 32
# mpiexec $flag  -n 64 $procname
# ./ipdps18 64
done
# mpiexec $flag  -n 16 $procname
# mpiexec $flag -n 32 $procname
# mpiexec $flag  -n 64 $procname
# mpiexec $flag  -n 64 ./build/test/allreduce

# procname=./build/test/allgather
# mpirun --mca btl self,vader,openib -np 64 $procname
# mpiexec -n 64 $procname


# export MV2_USE_SHMEM_COLL=1
# export MV2_ENABLE_TOPO_AWARE_COLLECTIVES=1
# export MV2_USE_TOPO_AWARE_ALLREDUCE=1
# export MV2_ENABLE_SOCKET_AWARE_COLLECTIVES=1
# procname=/public1/home/sc94715/software/osu-micro-benchmarks-6.2/c/mpi/collective/osu_allreduce -m 4096:268435456 -c
# export UCX_TLS=rc,ud,sm,self
# export FI_PROVIDER=mlx
# procname=/public1/home/sc94715/pjt/build/test/allreduce
# mpirun -np 1024 $procname
# srun -N 16 -n 1024 /public1/home/sc94715/software/osu-micro-benchmarks-6.2/c/mpi/collective/osu_allreduce -m 4096:268435456 -c
# mpirun -np 64 $procname
# mpiexec  -mca coll_hcoll_enable 1 -x HCOLL_MAIN_IB=mlx4_0:1 -n 1024 $procname
# srun -N 16 -n 1024 ./build/test/allreduce
# mpirun --mca btl self,vader,openib -np 64 $procname
# mpiexec -n 64 $procname

#mpirun --mca btl self,vader,openib -np 64 bcast
#lscpu
#numactl -H
