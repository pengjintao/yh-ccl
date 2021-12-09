#!/bin/bash
# sleep 3
# for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden/=2))
for((noden=$GLEX_COLL_NODEN_MIN;noden<=$GLEX_COLL_NODEN_MAX;noden*=2))
# for((noden=$GLEX_COLL_NODEN_MAX;noden>=$GLEX_COLL_NODEN_MIN;noden-=10))
do
    # noden=`expr 1 + $noden`
    export  GLEX_COLL_PROCN=`expr $GLEX_COLL_PPN \* $noden`
        for((loopn=0;loopn<1;loopn++))
        do 
            echo "---------------------------noden=$noden----------------------------------"
            yhrun -N $noden -n $GLEX_COLL_PROCN  ./build/test/$procName 
            wait
            sleep 4
        done
done
