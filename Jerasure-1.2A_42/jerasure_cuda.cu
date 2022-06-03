extern "C" 
{
  #include "galois.h"
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <stdbool.h>
  #include <time.h>
}
__constant__ long bitmatrix_constant[65536 / sizeof(long)];

/* First must to know, the basic calculate unit is packetsize * w bytes (I called it packetw), and to calculate one of such unit,
 * need to calculate w packets. to calculate one packet, need blocks_per_packet blocks, and one blocks need threads_per_block threads.
 * to loacte the data and coding pos, there are 3 shifts: shift for packetw, shift for packet and shift for thread.
 * and these shifts can be calculated as: index_packetw * packetsize_long * w, index_packet * packetsize_long and index_thread.
 * the first part of kernel is to calculate the parameters that mentioned before.
 */
__global__ void smpe(int k, int w, int coding_num, long *data, long *coding, int blocks_per_packet
    , int threads_per_block, int size_long, int packetsize_long)
{
  int packet_num = blockIdx.x / blocks_per_packet;
  int index_thread = (blockIdx.x - packet_num * blocks_per_packet) * blockDim.x + threadIdx.x;
  int index_packet = packet_num % w;
  int index_packetw = packet_num / w;
  int index_bitmatrix;
  long *coding_ptr, *data_ptr;
  long temp;
  int i, j;

  // add all three shifts
  coding_ptr = coding + coding_num * size_long + blockIdx.x * blockDim.x + threadIdx.x;
  // to reduce redudant calculate, first add shift for packetw and shift for thread
  data_ptr = data + index_packetw * packetsize_long * w + index_thread;
  index_bitmatrix = coding_num * k * w * w + index_packet * k * w;
  temp = 0;
  for (i = 0; i < k; i++) {
    __shared__ long data_shared[49152 / sizeof(long)];
    
    for (j = 0; j < w; j++)
      data_shared[j * blockDim.x + threadIdx.x] = data_ptr[i * size_long + j * packetsize_long];
    __syncthreads();
    for (j = 0; j < w; j++) {        
        temp ^= bitmatrix_constant[index_bitmatrix] & data_shared[j * blockDim.x + threadIdx.x];
        index_bitmatrix++;
    }
    __syncthreads();
  }
  *coding_ptr = temp;
}

extern "C" {
void jerasure_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  int threads_per_block, blocks_per_packet, blocks_num;
  static int prev_k = -1, prev_m = -1, prev_size = -1;
  static long *data_device, *coding_device;
  long *bitmatrix_long_temp;
  size_t size_free, size_total;
  int i;

  if (!(k == prev_k && m == prev_m && size == prev_size)) {
    if (data_device) {
        cudaFree(&data_device);
        cudaFree(&coding_device);
    }
    cudaMemGetInfo(&size_free, &size_total);
    if ((long)size * (k + m) > size_free) {
      fprintf(stderr, "buffer need %ld bytes, but GPU can only allocate %ld bytes\n", (long) size * (k + m), (long) free);
      exit(EXIT_FAILURE);
    }
    if (w * 4096 > 49152) {
      fprintf(stderr, "need shared memory %d MB(w * 4096), but only can support %d MB\n", w * 4, 49152 / 1024);
      exit(EXIT_FAILURE);
    }
    
    bitmatrix_long_temp = (long *) malloc(k * m * w * w * sizeof(long));
    for (i = 0; i < k * m * w * w; i++)
      if (bitmatrix[i] == 1)
        bitmatrix_long_temp[i] = 0xFFFFFFFFFFFFFFFF;
      else
        bitmatrix_long_temp[i] = 0;
    cudaMemcpyToSymbol(bitmatrix_constant, bitmatrix_long_temp, k * m * w * w * sizeof(long));
    free(bitmatrix_long_temp);
    
    cudaMalloc(&data_device, (long) size * k);
    cudaMalloc(&coding_device, (long) size * m);
    prev_k = k;
    prev_m = m;
    prev_size = size;
  }

  // by doing following calculation, guarantee every thread in blocks is used
  threads_per_block = packetsize / sizeof(long) < 512 ? packetsize / sizeof(long) : 512;
  if (packetsize < 512)
    blocks_per_packet = 1;
  else {
    blocks_per_packet = 1;
    while (true) {
      if (blocks_per_packet * 512 < packetsize / sizeof(long)) {
        blocks_per_packet++;
      }
      else if (blocks_per_packet * 512 == packetsize / sizeof(long))
        break;
      else {
        if ((size / sizeof(long)) % blocks_per_packet == 0) {
          threads_per_block = (packetsize / sizeof(long)) / blocks_per_packet;
          break;
        } else {
          blocks_per_packet++;
        }
      }
    }
  } 
  blocks_num = size / packetsize * blocks_per_packet;
  
  for (i = 0; i < k; i++)
    cudaMemcpy(data_device + i * size / sizeof(long), data_ptrs[i], size, cudaMemcpyHostToDevice);
  for (i = 0; i < m; i++)
    smpe<<<blocks_num, threads_per_block>>>(k, w, i, data_device, coding_device, blocks_per_packet, threads_per_block
      , size / sizeof(long), packetsize / sizeof(long));
  for (i = 0; i < m; i++)
    cudaMemcpy(coding_ptrs[i], coding_device + i * size / sizeof(long), size, cudaMemcpyDeviceToHost);
}
}
