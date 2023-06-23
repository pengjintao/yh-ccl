# yhccl is a high performance collective library for shared memory node

Running Example:
OSU All-reduce

CPU: AMD Ryzen 7 6800H
Open MPI 4.1.3
Disable YHCCL
 export OMPI_MCA_coll_yhccl_priority=0
 mpiexec -n 8 ./osu_allreduce -m 65536:268435456
 
![image](https://github.com/pengjintao/yh-ccl/assets/20380444/ae0e8a7d-a1e7-4092-aef4-27e60336672c)

Enable YHCCL
 export OMPI_MCA_coll_yhccl_priority=100
mpiexec -n 8 ./osu_allreduce -m 65536:268435456

![image](https://github.com/pengjintao/yh-ccl/assets/20380444/dced4a85-ced4-45dc-8729-38c6a4dd0ece)
