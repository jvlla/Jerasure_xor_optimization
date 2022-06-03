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
__global__ void gmpe(int k, int w, int coding_num, long *data, long *coding, int blocks_per_packet
    , int threads_per_block, int size_long, int packetsize_long, cudaTextureObject_t bitmatrix_texture)
{
  int packet_num = blockIdx.x / blocks_per_packet;
  int index_thread = (blockIdx.x - packet_num * blocks_per_packet) * blockDim.x + threadIdx.x;
  int index_packet;
  int index_packetw = packet_num;
  int index_bitmatrix;
  long *coding_ptr, *data_ptr;
  long temp;
  int i, j;

  // to reduce redudant calculate, first add shift for packetw and shift for thread
  data_ptr = data + index_packetw * packetsize_long * w + index_thread;
  index_bitmatrix = coding_num * k * w * w;
  
  for (index_packet = 0;index_packet < w; index_packet++) {
    // add all three shifts
    coding_ptr = coding + coding_num * size_long + index_packetw * packetsize_long * w + index_packet * packetsize_long + index_thread;
    
    temp = 0;
    for (i = 0; i < k; i++) {
      for (j = 0; j < w; j++) {
        if (tex1D<signed>(bitmatrix_texture, index_bitmatrix)) {
          temp ^= data_ptr[i * size_long + j * packetsize_long];
        }
        index_bitmatrix++;
      }
    }
    *coding_ptr = temp;
  }
}

extern "C" {
void jerasure_bitmatrix_encode(int k, int m, int w, int *bitmatrix,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  int threads_per_block, blocks_per_packet, blocks_num;
  static int prev_k = -1, prev_m = -1, prev_size = -1;
  static long *data_device, *coding_device;
  static int *bitmatrix_prev;
  cudaChannelFormatDesc channelDesc;
  static cudaArray *cuArray;
  struct cudaResourceDesc resDesc;
  struct cudaTextureDesc texDesc;
  static cudaTextureObject_t bitmatrix_texture = 0;
  size_t size_free, size_total;
  int i;

  if (!(k == prev_k && m == prev_m && size == prev_size && bitmatrix_prev == bitmatrix)) {
    if (data_device) {
      cudaFree(data_device);
      cudaFree(coding_device);
      cudaDestroyTextureObject(bitmatrix_texture);
      cudaFreeArray(cuArray);
    }
    
    cudaMemGetInfo(&size_free, &size_total);
    if ((long)size * (k + m) > size_free) {
      fprintf(stderr, "buffer need %ld bytes, but GPU can only allocate %ld bytes\n", (long) size * (k + m), (long) size_free);
      exit(EXIT_FAILURE);
    }
    
    cudaMalloc(&data_device, (long) size * k);
    cudaMalloc(&coding_device, (long) size * m);
    
    // copy bitmatrix to texture memory
    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindSigned);
    cudaMallocArray(&cuArray, &channelDesc, k * m * w * w, 0);
    cudaMemcpy2DToArray(cuArray, 0, 0, bitmatrix, sizeof(int) * k * m * w * w, sizeof(int) * k * m * w * w, 1, cudaMemcpyHostToDevice);

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&bitmatrix_texture, &resDesc, &texDesc, NULL);
    
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
  blocks_num = size / (packetsize * w) * blocks_per_packet;
  
  for (i = 0; i < k; i++)
    cudaMemcpy(data_device + i * size / sizeof(long), data_ptrs[i], size, cudaMemcpyHostToDevice);
  for (i = 0; i < m; i++)
    gmpe<<<blocks_num, threads_per_block>>>(k, w, i, data_device, coding_device, blocks_per_packet, threads_per_block
      , size / sizeof(long), packetsize / sizeof(long), bitmatrix_texture);
  for (i = 0; i < m; i ++)
    cudaMemcpy(coding_ptrs[i], coding_device + i * size / sizeof(long), size, cudaMemcpyDeviceToHost);
}
}
