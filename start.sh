#!/bin/sh

# module load cmake/3.18.0-rc2-fenggl-public3 
# module unload mpich
# module unload mpi
# module unload gcc
#cd /public3/home/sc53841/neighbor-collective
#cd /public3/home/sc53841/osu-micro-benchmarks-5.9/
# cd /public3/home/sc53841/yhccl

function llvm(){
source /public3/home/sc53841/software/llvm-project-llvmorg-14.0.6/llvm/env.sh
}
function rb(){
	cd ~/yhccl
		rm CMakeCache.txt
		cmake .
		make clean
		make -j 12
		make
}
function osu()
{
	cd  /public3/home/sc53841/osu-micro-benchmarks-5.9/mpi/collective
}
function osur()
{
	osu
		mpiexec -n 8 ./osu_allreduce -m 1:268435456
}
function dnn()
{
	module add python/3.6.5-torch-cjj-public3
		cd ~/software/horovod
}

function ompi_install()
{
	cd /public3/home/sc53841/software/ompi_config/ompi/mca/coll/yhccl   
		make -j 8
		make 
		make install
}
function ompi_yhccl(){
	module unload gcc
		module load gcc/11.1.0-jy-public3
		module unload mpi
		export mvapichdir=/public3/home/sc53841/software/ompi_build
		export PATH=${mvapichdir}/bin:$PATH
		export LD_LIBRARY_PATH=${mvapichdir}/lib:$LD_LIBRARY_PATH
		export C_INCLUDE_PATH=${mvapichdir}/include:$C_INCLUDE_PATH
		export CPLUS_INCLUDE_PATH=${mvapichdir}/include:$CPLUS_INCLUDE_PATH

}
function mvapich(){
	module unload mpi
#		module unload gcc
		export CXXFLAGS="-std=c++11 -fpermissive"
#		module load intel/2022.1
		# export mvapichdir=/public3/home/sc53841/software/mvapich
		export mvapichdir=/public1/home/sc94715/software/mvapich2-2.3.7-1/build
		export PATH=${mvapichdir}/bin:$PATH
		export LD_LIBRARY_PATH=${mvapichdir}/lib:$LD_LIBRARY_PATH
		export C_INCLUDE_PATH=${mvapichdir}/include:$C_INCLUDE_PATH
		export CPLUS_INCLUDE_PATH=${mvapichdir}/include:$CPLUS_INCLUDE_PATH
}
function impi(){
	module load mpi/intel/2022.1
# 	module unload mpi
# 		module unload gcc
# #module load mpi/intel/17.0.7-public3
# #		module load mpi/intel/2022.1
# 		module load intel/2022.1
# 		source ~/intel/oneapi/setvars.sh
#         module unload gcc
# 		module load gcc/11.1.0-jy-public3

}
function ompi(){
	module unload mpi
	module unload gcc
	module load gcc/7.3.0-kd
	module load mpi/openmpi/4.1.1-gcc7.3.0

}
function ci(){
	cd /public3/home/sc53841/yhccl
		rm CMakeCache.txt
		cmake .
		make clean
		make -j 12
		make
}
function papi(){
	export PAPI_DIR=/public3/home/sc53841/papi-6.0.0/build/
		export PATH=${PAPI_DIR}/bin:$PATH
		export LD_LIBRARY_PATH=${PAPI_DIR}/lib:$LD_LIBRARY_PATH
		export C_INCLUDE_PATH=${PAPI_DIR}/include:$C_INCLUDE_PATH
		export CPLUS_INCLUDE_PATH=${PAPI_DIR}/include:$CPLUS_INCLUDE_PATH
}

function mpich(){
	# module unload mpi
	# 	module unload gcc
	# 	module add mpich/3.1.4_gcc8.1.0-wjl-public3
	export ucxdir=/public1/home/sc94715/software/ucx/build/
	export PATH=${ucxdir}/bin/:$PATH
	export LD_LIBRARY_PATH=${ucxdir}/lib/:$LD_LIBRARY_PATH
	export C_INCLUDE_PATH=${ucxdir}/include:$C_INCLUDE_PATH
	export CPLUS_INCLUDE_PATH=${ucxdir}/include:$CPLUS_INCLUDE_PATH
		export mvapichdir=/public1/home/sc94715/software/mpich-4.1/build
		export PATH=${mvapichdir}/bin:$PATH
		export LD_LIBRARY_PATH=${mvapichdir}/lib:$LD_LIBRARY_PATH
		export C_INCLUDE_PATH=${mvapichdir}/include:$C_INCLUDE_PATH
		export CPLUS_INCLUDE_PATH=${mvapichdir}/include:$CPLUS_INCLUDE_PATH
}
function brun(){

	cd /public3/home/sc53841/yhccl
		make clean
		make -j8
		make -j8
		make
		rm slurm*
		sbatch ./sub.sh
		for((i=0;i<1000;i++))
			do
				sleep 1
					squeue
					cat slurm*
					done
}
function brun_NX(){

	cd /public1/home/scfa0319/pjt
		rm ./build/test/allreduce
		make clean
		make -j8
		make -j8
		make
		rm slurm*
		sbatch ./sub_NX.sh
		for((i=0;i<1000;i++))
			do
				sleep 1
					squeue
					cat slurm*
					done
}
function ucx()
{
	export ucxdir=/public1/home/sc94715/software/ucx/build/
	export PATH=${ucxdir}/bin/:$PATH
	export LD_LIBRARY_PATH=${ucxdir}/lib/:$LD_LIBRARY_PATH
	export C_INCLUDE_PATH=${ucxdir}/include:$C_INCLUDE_PATH
	export CPLUS_INCLUDE_PATH=${ucxdir}/include:$CPLUS_INCLUDE_PATH

}
function conda(){
    module load anaconda/3-Python3.7.4-fenggl-public3
}
export CC=mpicc
export CXX=mpicxx
