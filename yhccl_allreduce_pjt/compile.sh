#/bin/bash
rm ./build/test/allreduce
make clean
make yhccl -j 16 && make allreduce  
export noden=100
# yhbatch -N $noden -x `cat ./th2-exclude.txt` ./batch-allreduce.sh