#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <limits>

#include "cuda_utils.cuh"
#include "welford.h"

namespace megalodon {
namespace reduce {

template <typename T>
struct SumOp {
  static constexpr T kIdentityElement = T(0);

  __inline__ __device__ T operator()(T lhs, T rhs) const { return lhs + rhs; }
};

template <typename T>
struct SumOp<utils::WelfordData<T>> {
  static constexpr utils::WelfordData<T> kIdentityElement = {};

  __inline__ __device__ utils::WelfordData<T> operator()(
      utils::WelfordData<T> lhs, utils::WelfordData<T> rhs) const {
    return lhs + rhs;
  }
};

template <typename T>
struct MaxOp {
  static constexpr T kIdentityElement = std::numeric_limits<T>::lowest();

  __inline__ __device__ T operator()(T lhs, T rhs) const {
    return c10::cuda::compat::max(lhs, rhs);
  }
};

template <typename T>
struct MinOp {
  static constexpr T kIdentityElement = std::numeric_limits<T>::max();

  __inline__ __device__ T operator()(T lhs, T rhs) const {
    return c10::cuda::compat::min(lhs, rhs);
  }
};

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T WarpReduce(T x, ReduceOp reduce_op = SumOp<T>()) {
#pragma unroll
  for (int64_t offset = (cuda_utils::kWarpSize >> 1); offset > 0;
       offset >>= 1) {
    x = reduce_op(x, cuda_utils::WarpShflDown(x, offset));
  }
  return x;
}

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T BlockReduce(T x, T* shm,
                                    ReduceOp reduce_op = SumOp<T>()) {
  if (blockDim.x == cuda_utils::kWarpSize) {
    return WarpReduce(x, reduce_op);
  }
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % cuda_utils::kWarpSize;
  const int64_t wid = tid / cuda_utils::kWarpSize;
  const int64_t num_warps = blockDim.x / cuda_utils::kWarpSize;
  x = WarpReduce(x, reduce_op);
  __syncthreads();
  if (lid == 0) {
    shm[wid] = x;
  }
  __syncthreads();
  x = tid < num_warps ? shm[tid] : ReduceOp::kIdentityElement;
  if (wid == 0) {
    x = WarpReduce(x, reduce_op);
  }
  return x;
}

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T WarpAllReduce(T x, ReduceOp reduce_op = SumOp<T>()) {
#pragma unroll
  for (int64_t offset = (cuda_utils::kWarpSize >> 1); offset > 0;
       offset >>= 1) {
    x = reduce_op(x, cuda_utils::WarpShflXor(x, offset));
  }
  return x;
}

template <typename T, class ReduceOp = SumOp<T>>
__inline__ __device__ T BlockAllReduce(T x, T* shm,
                                       ReduceOp reduce_op = SumOp<T>()) {
  if (blockDim.x == cuda_utils::kWarpSize) {
    return WarpAllReduce(x, reduce_op);
  }
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % cuda_utils::kWarpSize;
  const int64_t wid = tid / cuda_utils::kWarpSize;
  const int64_t num_warps = blockDim.x / cuda_utils::kWarpSize;
  x = WarpReduce(x, reduce_op);
  __syncthreads();
  if (lid == 0) {
    shm[wid] = x;
  }
  __syncthreads();
  x = lid < num_warps ? shm[lid] : ReduceOp::kIdentityElement;
  x = WarpAllReduce(x, reduce_op);
  return x;
}

}  // namespace reduce
}  // namespace megalodon
