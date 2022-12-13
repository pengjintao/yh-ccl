#!/bin/bash
###
 # @Author: pengjintaoHPC 1272813056@qq.com
 # @Date: 2022-06-11 15:08:48
 # @LastEditors: pengjintaoHPC 1272813056@qq.com
 # @LastEditTime: 2022-06-12 20:03:31
 # @FilePath: \yhccl\batch-allreduce.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# sleep 3
# for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden/=2))
GLEX_COLL_NODEN_MIN=$noden
GLEX_COLL_NODEN_MAX=$noden

# for((ppn=8;ppn<=48;ppn+=40))

for((ppn=24;ppn<=24;ppn+=16))
do
    GLEX_COLL_PPN=$ppn
    for((n=$GLEX_COLL_NODEN_MAX;n>=$GLEX_COLL_NODEN_MIN;n/=2))
    # for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden-=10))
    do
        # noden=`expr 1 + $noden`
        export  GLEX_COLL_PROCN=`expr $GLEX_COLL_PPN \* $n`
            for((loopn=0;loopn<1;loopn++))
            do 
                # echo "------------MPI-----------noden=$n----ppn=$GLEX_COLL_PPN------------------------------"
                # export LD_PRELOAD=""
                # #yhrun -N $n -n $GLEX_COLL_PROCN  ./build/test/baidu_allreduce
                #  yhrun -N $n -n $GLEX_COLL_PROCN  /BIGDATA1/nudt_jliu_1/pjt/osu-micro-benchmarks-5.9/mpi/collective/osu_allreduce -m 134217728:134217728
                # wait
                # sleep 4
                # echo "------------YHCCL-----------noden=$n----ppn=$GLEX_COLL_PPN------------------------------"
                # export LD_PRELOAD="/BIGDATA1/nudt_jliu_1/pjt/yhccl-build/build/lib/libyhccl.so"
                # #yhrun -N $n -n $GLEX_COLL_PROCN  ./build/test/baidu_allreduce
                #  yhrun -N $n -n $GLEX_COLL_PROCN  /BIGDATA1/nudt_jliu_1/pjt/osu-micro-benchmarks-5.9/mpi/collective/osu_allreduce -m 134217728:134217728
                # wait
                # sleep 4

               echo "-----------------------noden=$n----ppn=$GLEX_COLL_PPN------------------------------"
                yhrun -N $n -n $GLEX_COLL_PROCN  ./build/test/allreduce 
                wait
                sleep 4
            done
    done
done

    # GLEX_COLL_PPN=24
    # # for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden/=2))
    # for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden-=10))
    # do
    #     # noden=`expr 1 + $noden`
    #     export  GLEX_COLL_PROCN=`expr $GLEX_COLL_PPN \* $noden`
    #         for((loopn=0;loopn<1;loopn++))
    #         do 
    #             echo "---------------mpi=pmix--------------noden=$noden----ppn=$GLEX_COLL_PPN------------------------------"
    #             yhrun -N $noden -n $GLEX_COLL_PROCN  ./build/test/allreduce
    #             wait
    #             sleep 4
    #         done
    # done