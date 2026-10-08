// Connected components of an undirected graph by GPU union-find (ECL-CC style): on return every node
// holds its component's smallest node.
//
// One thread per edge finds the roots of both ends (pointer chasing, then
// compressing the ends' own entries) and hooks the larger root under the
// smaller with an atomic compare-and-swap. A swap that loses a race has
// learned the root's new parent, which is smaller, and goes on from there. A
// final pass points every node at its root. A root is only ever hooked under a
// smaller node, so the component's smallest node is never hooked: whatever
// order the threads run in, it is the root everyone ends at.
//
// Built ahead of time for every GPU generation from Turing (sm_75) to
// Blackwell (sm_120), with compute_75 PTX for newer ones (compile_union_find.sh).
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

std::string last_error;

__device__ __forceinline__ int root_of(volatile int* parent, int x) {
  int p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  return x;
}

__global__ void hook(const int* __restrict__ rows, const int* __restrict__ cols, long long edges, int* parent) {
  volatile int* shared_parent = parent;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long k = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; k < edges; k += stride) {
    const int u = rows[k], v = cols[k];
    int a = root_of(shared_parent, u), b = root_of(shared_parent, v);
    // Compress only a node that is not a root: a root's entry belongs to the
    // hooks below, and any ancestor is a valid parent.
    if (a != u) shared_parent[u] = a;
    if (b != v) shared_parent[v] = b;
    while (a != b) {
      const int high = max(a, b), low = min(a, b);
      const int old = atomicCAS(parent + high, high, low);
      if (old == high) break;  // hooked
      a = old;                 // lost the race: high's parent is now old
      b = low;
    }
  }
}

__global__ void flatten(int* parent, long long nodes) {
  volatile int* shared_parent = parent;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < nodes; i += stride) {
    shared_parent[i] = root_of(shared_parent, static_cast<int>(i));
  }
}

int failed(const char* where, cudaError_t status) {
  last_error = std::string(where) + ": " + cudaGetErrorString(status);
  return 1;
}

long long blocks_for(long long work, int threads) {
  const long long blocks = (work + threads - 1) / threads;
  return blocks < (1LL << 20) ? blocks : (1LL << 20);  // grid-stride beyond this
}

}  // namespace

extern "C" {

int tg_union_find_abi_version() { return 1; }

const char* tg_union_find_error() { return last_error.c_str(); }

// rows, cols: the edges (device, int32); parent: 0 .. nodes - 1 on entry (device, int32), each node's
// component label on return. Runs on `stream` (a cudaStream_t). Returns 0, or 1 with tg_union_find_error().
int tg_union_find(const int32_t* rows, const int32_t* cols, int64_t edges, int32_t* parent, int64_t nodes,
                  void* stream) {
  if (edges <= 0 || nodes <= 0) return 0;
  const auto on = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  hook<<<blocks_for(edges, threads), threads, 0, on>>>(rows, cols, edges, parent);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return failed("hook", status);
  flatten<<<blocks_for(nodes, threads), threads, 0, on>>>(parent, nodes);
  status = cudaGetLastError();
  if (status != cudaSuccess) return failed("flatten", status);
  return 0;
}

}  // extern "C"
