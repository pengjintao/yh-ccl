#include <iostream>
#include "yhccl_contexts.h"

using namespace std;
int main()
{
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "starts" << endl;
    }
    // puts("INIT");
    pjtccl_contexts ccl_ctx;
    ccl_ctx.init(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}