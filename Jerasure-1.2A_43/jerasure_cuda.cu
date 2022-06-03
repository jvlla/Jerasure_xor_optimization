extern "C" 
{
  #include "galois.h"
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <stdbool.h>
  #include <time.h>
}

/* First must to know, the basic calculate unit is packetsize * w bytes (I called it packetw), and to calculate one of such unit,
 * need to calculate w packets. to calculate one packet, need blocks_per_packet blocks, and one blocks need threads_per_block threads.
 * to loacte the data and coding pos, there are 3 shifts: shift for packetw, shift for packet and shift for thread.
 * and these shifts can be calculated as: index_packetw * packetsize_long * w, index_packet * packetsize_long and index_thread.
 * the first part of kernel is to calculate the parameters that mentioned before.
 */
__global__ void kernel(int k, int w, int coding_num, long *data, long *coding, int blocks_per_packet
    , int threads_per_block, int size_long, int packetsize_long, int * bitmatrix)
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
    for (j = 0; j < w; j++) {
        if (bitmatrix[index_bitmatrix]) {
          temp ^= data_ptr[i * size_long + j * packetsize_long];
        }
        index_bitmatrix++;
    }
  }
  *coding_ptr = temp;
}

extern "C" {
void jerasure_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  int threads_per_block, blocks_per_packet, blocks_num;
  static int prev_k = -1, prev_m = -1, prev_size = -1;
  static int *bitmatrix_device;
  static long *data_device, *coding_device;
  size_t size_free, size_total;
  static int stream_num;
  static cudaStream_t *streams;
  static int shift_num;
  static int shifts[2];
  int i, j, packetw_shift;

  if (!(k == prev_k && m == prev_m && size == prev_size)) {
    if (data_device) {
      cudaFree(&data_device);
      cudaFree(&coding_device);
      cudaFree(&bitmatrix_device);
      for (i = 0; i < stream_num; i++)
        cudaStreamDestroy(streams[i]);
      free(streams);
    }
    cudaMemGetInfo(&size_free, &size_total);
    if ((long)size * (k + m) > size_free) {
      fprintf(stderr, "buffer need %ld bytes, but GPU can only allocate %ld bytes\n", (long) size * (k + m), (long) free);
      exit(EXIT_FAILURE);
    }
    
    cudaMalloc(&data_device, (long) size * k);
    cudaMalloc(&coding_device, (long) size * m);
    cudaMalloc(&bitmatrix_device, sizeof(int) * k * m * w * w);
    
    stream_num = min(8, size / (packetsize * w));
    if (stream_num < 8) {
      shift_num = 0;
      shifts[0] = shifts[1] = 1;
    } else {
      shifts[1] = size / (packetsize * w) / stream_num;
      shift_num = size / (packetsize * w) - stream_num * shifts[1];
      if (shift_num > 0) {
        shifts[0] = shifts[1] + 1;
      } else {
        shifts[0] = shifts[1];
      }
    }
    streams = (cudaStream_t *) malloc(stream_num * sizeof(cudaStream_t));
    for (i = 0; i < stream_num; i++)
      cudaStreamCreate(&streams[i]);
    
    prev_k = k;
    prev_m = m;
    prev_size = size;
  }

  // by doing following calculation, guarantee every thread in blocks is used
  threads_per_block = packetsize / sizeof(long) < 1024 ? packetsize / sizeof(long) : 1024;
  if (packetsize < 1024)
    blocks_per_packet = 1;
  else {
    blocks_per_packet = 1;
    while (true) {
      if (blocks_per_packet * 1024 < packetsize / sizeof(long)) {
        blocks_per_packet++;
      }
      else if (blocks_per_packet * 1024 == packetsize / sizeof(long))
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
  blocks_num = w * blocks_per_packet;
  
  cudaMemcpy(bitmatrix_device, bitmatrix, sizeof(int) * k * m * w * w, cudaMemcpyHostToDevice);
  for (i = 0; i < shift_num; i++) {
    packetw_shift = i * shifts[0] * packetsize * w;
    for (j = 0; j < k; j++)
      cudaMemcpy(data_device + (j * size + packetw_shift) / sizeof(long), data_ptrs[j] + packetw_shift, shifts[0] * packetsize * w, cudaMemcpyHostToDevice);
    for (j = 0; j < m; j++)
      kernel<<<shifts[0] * blocks_num, threads_per_block>>>(k, w, j, data_device + packetw_shift / sizeof(long), coding_device + packetw_shift / sizeof(long), blocks_per_packet, threads_per_block, size / sizeof(long), packetsize / sizeof(long), bitmatrix_device);
    for (j = 0; j < m; j++)
      cudaMemcpy(coding_ptrs[j] + packetw_shift, coding_device + (j * size + packetw_shift) / sizeof(long), shifts[0] * packetsize * w, cudaMemcpyDeviceToHost);
  }
  for (i = shift_num; i < stream_num; i++) {
    packetw_shift = (shift_num * shifts[0] + (i - shift_num) * shifts[1]) * packetsize * w;    
    for (j = 0; j < k; j++)
      cudaMemcpy(data_device + (j * size + packetw_shift) / sizeof(long), data_ptrs[j] + packetw_shift, shifts[1] * packetsize * w, cudaMemcpyHostToDevice);
    for (j = 0; j < m; j++)
      kernel<<<shifts[1] * blocks_num, threads_per_block>>>(k, w, j, data_device + packetw_shift / sizeof(long), coding_device + packetw_shift / sizeof(long), blocks_per_packet, threads_per_block, size / sizeof(long), packetsize / sizeof(long), bitmatrix_device);
    for (j = 0; j < m; j++)
      cudaMemcpy(coding_ptrs[j] + packetw_shift, coding_device + (j * size + packetw_shift) / sizeof(long), shifts[1] * packetsize * w, cudaMemcpyDeviceToHost);
  }
  
  cudaDeviceSynchronize();
}
}
