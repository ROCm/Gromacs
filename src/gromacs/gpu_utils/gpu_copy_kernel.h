#include "hip/hip_runtime.h"

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__device__ inline
void block_load_direct_striped(unsigned int flat_id,
                               float* block_input,
                               float* items)
{
    float* thread_iter = block_input + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * BlockSize];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__device__ inline
void block_load_direct_striped(unsigned int flat_id,
                               float* block_input,
                               float* items,
                               unsigned int valid)
{
    float* thread_iter = block_input + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
            items[item] = thread_iter[offset];
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__device__ inline
void block_store_direct_striped(unsigned int flat_id,
                                float* block_output,
                                float* items,
                                unsigned int valid)
{
    float* thread_iter = block_output + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
             thread_iter[offset] = items[item];
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__device__ inline
void block_store_direct_striped(unsigned int flat_id,
                                float* block_output,
                                float* items)
{
    float* thread_iter = block_output + flat_id;
    #pragma unroll
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
         thread_iter[item * BlockSize] = items[item];
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__launch_bounds__(BlockSize)
__global__ void kernel_copy(
  void* src_ptr,
  void* dst_ptr,
  unsigned int memoryElementNumber)
{
  constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

  const unsigned int flat_id = threadIdx.x;
  const unsigned int flat_block_id = blockIdx.x;
  const unsigned int block_offset = flat_block_id * items_per_block;
  const unsigned int number_of_blocks = gridDim.x;
  const unsigned int valid_in_last_block = memoryElementNumber - block_offset;

  float input_values[ItemsPerThread];

  float* input  = static_cast<float*>(src_ptr);
  float* output = static_cast<float*>(dst_ptr);

  if(flat_block_id == (number_of_blocks - 1)) // last block
  {
       block_load_direct_striped<BlockSize, ItemsPerThread>(
             flat_id,
             input + block_offset,
             input_values,
             valid_in_last_block
      );

      block_store_direct_striped<BlockSize, ItemsPerThread>(
          flat_id,
          output + block_offset,
          input_values,
          valid_in_last_block
      );
  }
  else
  {
      block_load_direct_striped<BlockSize, ItemsPerThread>(
          flat_id,
          input + block_offset,
          input_values
      );

      block_store_direct_striped<BlockSize, ItemsPerThread>(
          flat_id,
          output + block_offset,
          input_values
      );
  }
}
