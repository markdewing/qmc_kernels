
// Vector add example in C using MPI shared memory

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define RealType double
#define N (1000*1000*8)

double clock()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_usec * 1e-6 + (double)tv.tv_sec;
}

void vector_add(RealType *a, RealType *b, RealType *c, int start, int end)
{
  for (int i = start; i < end; i++) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv)
{

  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &nodes);

  size_t array_bytes = N * sizeof(RealType);
  if (rank == 0) {
    printf("Array size = %g GB\n",array_bytes/1e9);
  }

  // Create shared memory communicator on a single node
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);


  int window_size = 0;
  // Only need to allocate a non-zero buffer with one rank
  if (rank == 0) window_size = 3*N*sizeof(RealType);

  int disp = sizeof(RealType);

  MPI_Aint baseptr;
  MPI_Win win;
  // Allocated shared memory for all three arrays at once
  MPI_Win_allocate_shared(window_size, disp, MPI_INFO_NULL, node_comm, (void *)&baseptr, &win);

  // Get local address of shared buffer
  MPI_Aint node_size;
  int node_disp_unit;
  MPI_Aint node_ptr;
  MPI_Win_shared_query(win, 0, &node_size, &node_disp_unit, &node_ptr);

  RealType *a = (RealType *)(node_ptr);
  RealType *b = (RealType *)(node_ptr + N*node_disp_unit);
  RealType *c = (RealType *)(node_ptr + 2*N*node_disp_unit);

  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 1.0*i;
  }

  MPI_Win_fence(0, win);

  // Divide up the work - compute ranges for each node
  int items_per_rank = N/nodes;
  int istart = rank*items_per_rank;
  int iend = istart + items_per_rank;

  if (rank == nodes-1) iend = N;

  double start = clock();

  vector_add(a, b, c, istart, iend);

  MPI_Win_fence(0, win);
  double end = clock();

  double elapsed = end - start;
  double elapsed_ns = elapsed * 1e9;

  if (c[N-1] != N) {
    printf("incorrect c[N-1] = %g (should be %d)\n",c[N-1],N);
  }

  size_t read_bytes = 2 * N * sizeof(RealType);
  size_t write_bytes = N * sizeof(RealType);
  size_t rw_bytes = read_bytes + write_bytes;

  double bw = rw_bytes * 1.0 / elapsed_ns;

  if (rank == 0) {
    printf(" Elapsed time = %g ns\n",elapsed_ns);
    printf(" Elapsed time = %g ns per element\n",elapsed_ns/N);
    printf(" BW (GB/s) = %g\n",bw);

  }

  MPI_Finalize();

  return 0;
}
