
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#if !defined(N)
#define N 12 // unsorted array size
#endif

// Merges sorted chunks across MPI ranks in a tree pattern.
std::vector<int> treeReduction(std::vector<int> localData, int rank, int processCount, int localSize) {
    // Stride controls partner distance in each merge round.
    int stride = 1;
    // Current number of sorted items expected from each sender.
    int sortedBlockSize = localSize;

    // Keep merging until rank 0 has the full sorted array.
    while (stride < processCount) {
        // Receiver ranks for this stride.
        if (rank % (2 * stride) == 0) {
            int sender = rank + stride;
            if (sender < processCount) {
                // Receive sorted block from partner rank.
                std::vector<int> incoming(sortedBlockSize);
                MPI_Recv(incoming.data(), sortedBlockSize, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Append and merge two sorted halves in place.
                localData.insert(localData.end(), incoming.begin(), incoming.end());
                std::inplace_merge(localData.begin(), localData.begin() + sortedBlockSize, localData.end());
            }
        } else {
            // Sender ranks send once and exit.
            int receiver = rank - stride;
            MPI_Send(localData.data(), static_cast<int>(localData.size()), MPI_INT, receiver, 0, MPI_COMM_WORLD);
            break;
        }

        // Move to next tree level.
        sortedBlockSize *= 2;
        stride *= 2;
    }

    return localData;
}

// Creates and prints a random input array on rank 0.
std::vector<int> createArray() {
    std::cout << "Unsorted Array: [";
    std::vector<int> unsorted(N);
    for (int i = 0; i < N; ++i) {
        // Generate values in [0, 99].
        unsorted[i] = rand() % 100;
        std::cout << unsorted[i] << ", ";
    }
    std::cout << "]" << std::endl;
    return unsorted;
}

int main(int argc, char* argv[]) {
    // Start MPI runtime.
    MPI_Init(&argc, &argv);

    // Get current rank id and total number of ranks.
    int rank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    std::vector<int> unsorted;
    if (rank == 0) {
        // Only root creates the full input array.
        unsorted = createArray();
    }

    // Each rank handles an equal chunk.
    int localSize = N / processCount;
    std::vector<int> localVector(localSize);

    // Scatter chunks from rank 0 to all ranks.
    MPI_Scatter(unsorted.data(), localSize, MPI_INT, localVector.data(), localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort local chunk, then merge globally via tree reduction.
    std::sort(localVector.begin(), localVector.end());
    localVector = treeReduction(localVector, rank, processCount, localSize);

    if (rank == 0) {
        // Root prints final sorted output.
        std::cout << "Sorted Array: [";
        for (int x : localVector) std::cout << x << ", ";
        std::cout << "]" << std::endl;
    }
    // Shut down MPI runtime.
    MPI_Finalize();
    return 0;
}