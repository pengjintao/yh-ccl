#/bin/bash

module add cmake/3.20.2
export LD_LIBRARY_PATH=/BIGDATA1/nudt_jliu_1/pjt/yh-coll/GLEX_Coll_lib:$LD_LIBRARY_PATH
export  GLEX_COLL_PPN=2
export  procName=allreduce


python ./job-auto-cancel.py
# rm CMakeCache.txt
# cmake .
# make clean
make yhccl -j 4
make allreduce
rm slurm*
MODE=1
case $MODE in
    1   )
        echo " allreduce Testing"
        export  GLEX_COLL_NODEN_MAX=12
        export  GLEX_COLL_NODEN_MIN=12
        wait
        #-x `cat ./node-screening/excludeth2` 
        yhbatch  -N $GLEX_COLL_NODEN_MAX --time=0-0:20:20  ./batch-allreduce.sh
         ;;
esac
for((i=1;i<=1000;i++))
do
    echo "---------------------------------------------------------------------------------------"
    cat slurm* 
    sleep 2
    echo "---------------------------------------------------------------------------------------"
done
