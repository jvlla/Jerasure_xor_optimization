extern "C" 
{
  #include "galois.h"
  #include "jerasure.h"
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <stdio.h>
  #include <stdbool.h>
  #include <time.h>
}

__global__ void jerasure_bitmatrix_dotprod_cuda(int k, int w, int *bitmatrix_row, int *src_ids, int dest_id
  , long *data, long *coding, int size_long, int packetsize_long, int blocks_per_packet)
{
  int packet_num = blockIdx.x / blocks_per_packet;
  int index_thread = (blockIdx.x - packet_num * blocks_per_packet) * blockDim.x + threadIdx.x;
  int index_packet = packet_num % w;
  int index_packetw = packet_num / w;
  int index_bitmatrix;
  long *data_ptr, *coding_ptr;
  long temp;
  int i, j;
  
  // add all three shifts
  if (dest_id < k) {
    coding_ptr = data + dest_id * size_long + blockIdx.x * blockDim.x + threadIdx.x;
  } else {
    coding_ptr = coding + (dest_id - k) * size_long + blockIdx.x * blockDim.x + threadIdx.x;
  }
  // to reduce redudant calculate, first add shift for packetw and shift for thread
  index_bitmatrix = index_packet * k * w;
  temp = 0;
  for (i = 0; i < k; i++) {
    if (src_ids == NULL) {
      data_ptr = data + i * size_long + index_packetw * packetsize_long * w + index_thread;
    } else {
      if (src_ids[i] < k) {
        data_ptr = data + (src_ids[i]) * size_long + index_packetw * packetsize_long * w + index_thread;
      } else {
        data_ptr = coding + (src_ids[i] - k) * size_long + index_packetw * packetsize_long * w + index_thread;
      }
    }
    for (j = 0; j < w; j++) {
      if (bitmatrix_row[index_bitmatrix]) {
        temp ^= data_ptr[j * packetsize_long];
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
  static long prev_size = -1, once_size;
  static int prev_k = -1, prev_m = -1;
  static int *bitmatrix_device;
  static long *data_device, *coding_device;
  size_t size_free, size_total;
  int i, once_time;
  long copy_pos;

  if (!(k == prev_k && m == prev_m && size == prev_size)) {
    if (data_device) {
      cudaFree(&data_device);
      cudaFree(&coding_device);
      cudaFree(&bitmatrix_device);
    }
    cudaMemGetInfo(&size_free, &size_total);
    //size_free = 1000000;
    if ((long)size * (k + m) > size_free) {
      once_size = (size_free / (k + m)) / (w * packetsize) * (w * packetsize);
    } else {
      once_size = size;
    }
    
    cudaMalloc(&data_device, (long) once_size * k);
    cudaMalloc(&coding_device, (long) once_size * m);
    cudaMalloc(&bitmatrix_device, sizeof(int) * k * m * w * w);
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
        if ((once_size / sizeof(long)) % blocks_per_packet == 0) {
          threads_per_block = (packetsize / sizeof(long)) / blocks_per_packet;
          break;
        } else {
          blocks_per_packet++;
        }
      }
    }
  } 
  blocks_num = once_size / packetsize * blocks_per_packet;
  
  cudaMemcpy(bitmatrix_device, bitmatrix, sizeof(int) * k * m * w * w, cudaMemcpyHostToDevice);
  for (once_time = 0; once_time * once_size < size; once_time++) {
    copy_pos = once_time * once_size;
    long sub_size = ((once_time + 1) * once_size) <= size ? once_size : size - once_time * once_size;
    
    for (i = 0; i < k; i++)
      cudaMemcpy(data_device + i * once_size / sizeof(long), data_ptrs[i] + copy_pos, sub_size, cudaMemcpyHostToDevice);
    for (i = 0; i < m; i++)
      jerasure_bitmatrix_dotprod_cuda<<<blocks_num, threads_per_block>>>(k, w, bitmatrix_device+i*k*w*w, NULL
        , k+i, data_device, coding_device, once_size / sizeof(long), packetsize / sizeof(long), blocks_per_packet);
    for (i = 0; i < m; i++)
      cudaMemcpy(coding_ptrs[i] + copy_pos, coding_device + i * once_size / sizeof(long), sub_size, cudaMemcpyDeviceToHost);  
  }
}

int jerasure_bitmatrix_decode(int k, int m, int w, int *bitmatrix, int row_k_ones, int *erasures,
                            char **data_ptrs, char **coding_ptrs, int size, int packetsize)
{
  int i, once_time;
  long copy_pos;
  int *erased;
  int *decoding_matrix;
  int *dm_ids;
  int edd, edd_loop, *tmpids, lastdrive;
  static int prev_k = -1, prev_m = -1, prev_size = -1, once_size;
  int threads_per_block, blocks_per_packet, blocks_num;
  size_t size_free, size_total;
  static int *bitmatrix_device, *decoding_matrix_device;
  static long *data_device, *coding_device;
  static int *dm_ids_device, *tmpids_device;
  
  erased = jerasure_erasures_to_erased(k, m, erasures);
  if (erased == NULL) return -1;
  
  if (prev_k != k || prev_m != m || prev_size != size) {
    if (prev_k != 0) {
      cudaFree(&dm_ids_device);
      cudaFree(&decoding_matrix_device);
      cudaFree(&data_device);
      cudaFree(&coding_device);
      cudaFree(&bitmatrix_device);
      cudaFree(&tmpids_device);
    }
    cudaMemGetInfo(&size_free, &size_total);
    if ((long)size * (k + m) > size_free) {
      fprintf(stderr, "buffer need %ld bytes, but GPU can only allocate %ld bytes\n", (long) size * (k + m), (long) free);
      exit(EXIT_FAILURE);
    }
    if ((long)size * (k + m) > size_free) {
      once_size = (size_free / (k + m)) / (w * packetsize) * (w * packetsize);
    } else {
      once_size = size;
    }
    
    cudaMalloc(&dm_ids_device, sizeof(int) *k);
    cudaMalloc(&decoding_matrix_device, sizeof(int)*k*k*w*w);
    cudaMalloc(&data_device, (long)once_size * k);
    cudaMalloc(&coding_device, (long)once_size * m);
    cudaMalloc(&bitmatrix_device, sizeof(int) * k * m * w * w);
    cudaMalloc(&tmpids_device, sizeof(int) * k);
    
    prev_k = k;
    prev_m = m;
    prev_size = size;
  }

  /* See jerasure_matrix_decode for the logic of this routine.  This one works just like
     it, but calls the bitmatrix ops instead */

  lastdrive = k;
    
  edd = 0;
  for (i = 0; i < k; i++) {
    if (erased[i]) {
      edd++;
      lastdrive = i;
    } 
  }

  if (row_k_ones != 1 || erased[k]) lastdrive = k;
  
  dm_ids = NULL;
  decoding_matrix = NULL;
  
  if (edd > 1 || (edd > 0 && (row_k_ones != 1 || erased[k]))) {
    dm_ids = (int *) malloc(sizeof(int) * k);
    
    if (dm_ids == NULL) {
      free(erased);
      return -1;
    }
  
    decoding_matrix = (int *) malloc(sizeof(int)*k*k*w*w);
    if (decoding_matrix == NULL) {
      free(erased);
      free(dm_ids);
      return -1;
    }

    if (jerasure_make_decoding_bitmatrix(k, m, w, bitmatrix, erased, decoding_matrix, dm_ids) < 0) {
      free(erased);
      free(dm_ids);
      free(decoding_matrix);
      return -1;
    }
    cudaMemcpy(dm_ids_device, dm_ids, sizeof(int) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(decoding_matrix_device, decoding_matrix, sizeof(int)*k*k*w*w, cudaMemcpyHostToDevice);
  }
  
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
        if ((once_size / sizeof(long)) % blocks_per_packet == 0) {
          threads_per_block = (packetsize / sizeof(long)) / blocks_per_packet;
          break;
        } else {
          blocks_per_packet++;
        }
      }
    }
  } 
  blocks_num = once_size / packetsize * blocks_per_packet;
  
  cudaMemcpy(bitmatrix_device, bitmatrix, sizeof(int) * k * m * w * w, cudaMemcpyHostToDevice);
  for (once_time = 0; once_time * once_size < size; once_time++) {
    copy_pos = once_time * once_size;
    long sub_size = ((once_time + 1) * once_size) <= size ? once_size : size - once_time * once_size;;
    
    for (i = 0; i < k + m; i++)
      if (!erased[i]) {
        if (i < k) {
          cudaMemcpy(data_device + i * once_size / sizeof(long), data_ptrs[i] + copy_pos, sub_size, cudaMemcpyHostToDevice);
        } else {
          cudaMemcpy(coding_device + (i-k) * once_size / sizeof(long), coding_ptrs[i-k] + copy_pos, sub_size, cudaMemcpyHostToDevice);
        }
      }

    edd_loop = edd;
    for (i = 0; edd_loop > 0 && i < lastdrive; i++) {
      if (erased[i]) {
        jerasure_bitmatrix_dotprod_cuda<<<blocks_num, threads_per_block>>>(k, w, decoding_matrix_device+i*k*w*w, dm_ids_device
          , i, data_device, coding_device, once_size / sizeof(long), packetsize / sizeof(long), blocks_per_packet);
        edd_loop--;
      }
    }

    if (edd_loop > 0) {
      tmpids = (int *) malloc(sizeof(int) * k);
      for (i = 0; i < k; i++) {
        tmpids[i] = (i < lastdrive) ? i : i+1;
      }
      cudaMemcpy(tmpids_device, tmpids, sizeof(int) *k, cudaMemcpyHostToDevice);
      jerasure_bitmatrix_dotprod_cuda<<<blocks_num, threads_per_block>>>(k, w, bitmatrix_device, tmpids_device
        , lastdrive, data_device, coding_device, once_size / sizeof(long), packetsize / sizeof(long), blocks_per_packet);
      free(tmpids);
    }

    for (i = 0; i < m; i++) {
      if (erased[k+i]) {
        jerasure_bitmatrix_dotprod_cuda<<<blocks_num, threads_per_block>>>(k, w, bitmatrix_device+i*k*w*w, NULL
          , k+i, data_device, coding_device, once_size / sizeof(long), packetsize / sizeof(long), blocks_per_packet);
      }
    }

    for (i = 0; i < k + m; i++)
      if (erased[i]) {
        if (i < k) {
          cudaMemcpy(data_ptrs[i] + copy_pos, data_device + i * once_size / sizeof(long), sub_size, cudaMemcpyDeviceToHost);
        } else {
          cudaMemcpy(coding_ptrs[i-k] + copy_pos, coding_device + (i - k) * once_size / sizeof(long), sub_size, cudaMemcpyDeviceToHost);
        }
      }
  }
  free(erased);
  if (dm_ids != NULL) free(dm_ids);
  if (decoding_matrix != NULL) free(decoding_matrix);

  return 0;
}
}
